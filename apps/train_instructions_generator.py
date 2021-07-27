from dataloader import collate_fn
from dataloader import CraftingDataset
from dataloader import generate_vocab
from dataloader import load_pkl
from instructions_generator_model import InstructionsGeneratorModel
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader


def train_model(device,
                data_dir,
                model_save_dir='trained_model/instructions_generator.pt',
                epochs=15,
                learning_rate=0.001,
                batch_size=64,
                vectors_cache=None,
                embeded_dim=300,
                log_size=500):
  train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions = load_pkl(
      workdir=data_dir)

  vocab, vocab_weights = generate_vocab(
      all_instructions, device, workdir=data_dir, cache=vectors_cache)

  dataset = CraftingDataset(
      embeded_dim,
      train_states,
      train_inventories,
      train_actions,
      train_goals,
      train_instructions,
      vocab,
      cache=vectors_cache)
  data_loader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=0,
      pin_memory=True,
      collate_fn=collate_fn)

  instructions_generator = InstructionsGeneratorModel(device, len(vocab),
                                                      embeded_dim,
                                                      vocab_weights)
  instructions_generator.to(device)
  instructions_generator.train()

  criterion = nn.CrossEntropyLoss()
  parameters = filter(lambda p: p.requires_grad,
                      instructions_generator.parameters())
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)

  for epoch in range(epochs):
    all_losses = []
    running_loss = 0.0
    running_loss_count = 0

    for i, data in enumerate(data_loader, 0):
      grid_onehot, grid_embedding, inventory_embedding, _, goal_embedding, instructions, lengths = data

      grid_onehot = grid_onehot.to(device)
      grid_embedding = grid_embedding.to(device)
      goal_embedding = goal_embedding.to(device)
      inventory_embedding = inventory_embedding.to(device)

      instructions = instructions.to(device)

      predictions, decode_lengths, alphas, _ = instructions_generator(
          grid_embedding, grid_onehot, inventory_embedding, goal_embedding,
          instructions, lengths)

      targets = instructions[:, 1:]
      try:
        predictions = pack_padded_sequence(
            predictions, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)[0]

        lang_loss = criterion(predictions, targets)
        lang_loss += 1. * ((1. - alphas.sum(dim=1))**2).mean()
        clip_grad_norm_(parameters, max_norm=3)

        optimizer.zero_grad()

        lang_loss.backward()
        optimizer.step()

        running_loss += lang_loss.item()
        running_loss_count += 1

      except RuntimeError as error:
        print(error)

      if i % log_size == log_size - 1:
        all_losses.append(running_loss / running_loss_count)

        print('[%d, %5d] lang loss: %.3f' %
              (epoch + 1, i + 1, running_loss / running_loss_count))

        running_loss = 0.0
        running_loss_count = 0

  torch.save(instructions_generator.state_dict(), model_save_dir)
  print('Trained model saved at ', model_save_dir)
  return instructions_generator

# TODO(xingnan): Load these parameters from config file
DATA_DIR = '/usr/local/google/home/billzhou/Documents/dataset_split/'
VECTORS_CACHE = '/usr/local/google/home/billzhou/Documents/glove'

if __name__ == '__main__':
  if torch.cuda.is_available():
    print('using cuda')
    device = torch.device('cuda')
  else:
    print('using cpu')
    device = torch.device('cpu')

  train_model(
      device,
      DATA_DIR,
      vectors_cache=VECTORS_CACHE,
      model_save_dir='trained_model/tmp.pt')
