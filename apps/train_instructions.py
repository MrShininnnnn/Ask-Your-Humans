from args_loader import load_args
from dataloader import collate_fn
from dataloader import CraftingDataset
from dataloader import generate_vocab
from dataloader import load_pkl
from instruction_model import InstructionModel
from instructions_discriminate_model import InstructionsDiscriminateModel
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader


def train_model(device):
  args = load_args('Instructions')

  train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions = load_pkl(
      workdir=args.data_dir)

  vocab, vocab_weights = generate_vocab(
      all_instructions, device, workdir=args.data_dir, cache=args.vectors_cache)

  dataset = CraftingDataset(
      args.embeded_dim,
      train_states,
      train_inventories,
      train_actions,
      train_goals,
      train_instructions,
      vocab,
      cache=args.vectors_cache)
  data_loader = DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=0,
      pin_memory=True,
      collate_fn=collate_fn)

  print('Data load success.')

  #ir_generator = InstructionModel(device, args.embeded_dim)
  ir_generator = InstructionsDiscriminateModel(device, len(vocab),
                                               args.embeded_dim, vocab_weights)

  ir_generator.to(device)
  ir_generator.train()

  criterion = nn.CrossEntropyLoss()
  parameters = filter(lambda p: p.requires_grad,
                      ir_generator.parameters())
  optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

  for epoch in range(args.epochs):
    all_losses = []
    running_loss = 0.0
    running_loss_count = 0

    for i, data in enumerate(data_loader, 0):
      grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions, lengths = data

      grid_onehot = grid_onehot.to(device)
      grid_embedding = grid_embedding.to(device)
      goal_embedding = goal_embedding.to(device)
      inventory_embedding = inventory_embedding.to(device)

      instructions = instructions.to(device)

      #predictions = ir_generator(
      #    grid_embedding, grid_onehot, inventory_embedding, goal_embedding)
      predictions = ir_generator(
           grid_embedding, grid_onehot, inventory_embedding, goal_embedding, instructions, lengths)

      #prediction = torch.argmax(predictions, dim = 1)
      action = action.to(device, dtype=torch.int64)
      action = action.squeeze(1)
      #print(predictions)
      #print(action)
      #print(predictions.shape)
      #print(action.shape)

      try:

        loss = criterion(predictions, action)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_count += 1

      except RuntimeError as error:
        print(error)

      if i % args.log_size == args.log_size - 1:
        all_losses.append(running_loss / running_loss_count)

        print('[%d, %5d] lang loss: %.3f' %
              (epoch + 1, i + 1, running_loss / running_loss_count))

        running_loss = 0.0
        running_loss_count = 0

  torch.save(ir_generator.state_dict(), args.model_save_dir)
  print('Trained model saved at ', args.model_save_dir)
  return ir_generator


if __name__ == '__main__':
  if torch.cuda.is_available():
    print('using cuda')
    device = torch.device('cuda')
  else:
    print('using cpu')
    device = torch.device('cpu')

  train_model(device)
