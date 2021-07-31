from args_loader import load_args
from dataloader import collate_fn
from dataloader import CraftingDataset
from dataloader import generate_vocab
from dataloader import load_pkl
from instructions_generator_metrics import InstructionsGeneratorMetrics
from instructions_generator_model import InstructionsGeneratorModel
from imitation_learning_with_generative_language_model import ImitationLearningWithGenerativeLanguageModel
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score


def calculate_loss(criterion, predictions, targets, decode_lengths, alphas):
  predictions = pack_padded_sequence(
      predictions, decode_lengths, batch_first=True)[0]
  targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

  loss = criterion(predictions, targets)
  loss += 1. * ((1. - alphas.sum(dim=1))**2).mean()
  return loss


def train(device,
          epoch,
          train_data_loader,
          il_model,
          lstm_model,
          optimizer,
          lstm_optimizer,
          criterion,
          lstm_criterion,
          parameters,
          lstm_parameters,
          vocab,
          log_size=500,
          use_generative_language=True,
          summary_writer=None):
  il_model.train()
  lstm_model.train()
  metrics = InstructionsGeneratorMetrics(vocab, lstm_criterion)

  running_loss = []
  all_losses = []

  for i, data in enumerate(train_data_loader, 0):
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    instructions = instructions.to(device)
    action = action.to(device, dtype=torch.int64)
    action = action.squeeze(1)

    if use_generative_language:
      use_teacher_forcing = True if random.random() < 0.5 else False

      lstm_predictions, decode_lengths, alphas, lstm_hiddens = lstm_model(
          grid_embedding,
          grid_onehot,
          inventory_embedding,
          goal_embedding,
          instructions,
          lengths,
          use_teacher_forcing=use_teacher_forcing)
    else:
      lstm_hiddens = None

    predictions = il_model(
        grid_embedding,
        grid_onehot,
        inventory_embedding,
        goal_embedding,
        lstm_hiddens=lstm_hiddens)

    try:
      action_loss = criterion(predictions, action)
      clip_grad_norm_(parameters, max_norm=3)

      if use_generative_language:
        targets = instructions[:, 1:]
        lstm_loss = metrics.add(lstm_predictions, targets, decode_lengths,
                                alphas)
        clip_grad_norm_(lstm_parameters, max_norm=3)

        lstm_optimizer.zero_grad()
        total_loss = action_loss + lstm_loss
      else:
        total_loss = action_loss

      optimizer.zero_grad()
      total_loss.backward()

      optimizer.step()
      if use_generative_language:
        lstm_optimizer.step()

      running_loss.append(action_loss.item())

    except RuntimeError as error:
      print(error)

    if i % log_size == log_size - 1:
      all_losses.append(np.array(running_loss).mean())
      metrics.flush(epoch, i)

      print('[%d, %5d] train loss: %.3f' %
            (epoch + 1, i + 1, np.array(running_loss).mean()))

      running_loss = []

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/train',
                              np.array(all_losses).mean(), epoch + 1)


def validate(device,
             epoch,
             val_loader,
             il_model,
             lstm_model,
             criterion,
             vocab,
             log_size=500,
             use_generative_language=True,
             summary_writer=None):
  running_loss = []
  all_losses = []
  il_model.eval()
  lstm_model.eval()

  for idx, data in enumerate(val_loader, 0):
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    instructions = instructions.to(device)
    action = action.to(device, dtype=torch.int64)
    action = action.squeeze(1)

    with torch.no_grad():
      if use_generative_language:
        # If not using teacher, the loss is very high.
        _, _, _, lstm_hiddens = lstm_model(
            grid_embedding,
            grid_onehot,
            inventory_embedding,
            goal_embedding,
            instructions,
            lengths,
            use_teacher_forcing=False)
      else:
        lstm_hiddens = None

      predictions = il_model(
          grid_embedding,
          grid_onehot,
          inventory_embedding,
          goal_embedding,
          lstm_hiddens=lstm_hiddens)
      try:
        loss = criterion(predictions, action)
        running_loss.append(loss.item())

      except RuntimeError as error:
        print(error)

    if idx % log_size == log_size - 1:
      all_losses.append(np.array(running_loss).mean())

      print('[%d, %5d] valid loss: %.3f' %
            (epoch + 1, idx + 1, np.array(running_loss).mean()))

      running_loss = []

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/valid',
                              np.array(all_losses).mean(), epoch + 1)


def main():
  if torch.cuda.is_available():
    print('using cuda')
    device = torch.device('cuda')
  else:
    print('using cpu')
    device = torch.device('cpu')

  args = load_args('Instructions Generator')

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

  validation_split = 0.2
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))

  np.random.seed(123)
  np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  train_data_loader = DataLoader(
      dataset,
      batch_size=args.batch_size,
      num_workers=0,
      pin_memory=True,
      sampler=train_sampler,
      collate_fn=collate_fn)
  validation_data_loader = DataLoader(
      dataset,
      batch_size=args.batch_size,
      num_workers=0,
      pin_memory=True,
      sampler=valid_sampler,
      collate_fn=collate_fn)
  lstm_model = InstructionsGeneratorModel(device, vocab, args.embeded_dim,
                                          vocab_weights)
  # lstm_model.load_state_dict(torch.load(args.pretrained_instructions_generator))
  lstm_model.to(device)

  model = ImitationLearningWithGenerativeLanguageModel(args.embeded_dim)
  model.to(device)
  model.train()

  criterion = nn.CrossEntropyLoss().to(device)
  lstm_criterion = nn.CrossEntropyLoss().to(device)
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  lstm_parameters = filter(lambda p: p.requires_grad, lstm_model.parameters())
  optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
  lstm_optimizer = torch.optim.Adam(lstm_parameters, lr=args.learning_rate)

  writer = SummaryWriter(log_dir='runs/il_only'
                        ) if args.summary_writer else None

  for epoch in range(args.epochs):
    train(
        device,
        epoch,
        train_data_loader,
        model,
        lstm_model,
        optimizer,
        lstm_optimizer,
        criterion,
        lstm_criterion,
        parameters,
        lstm_parameters,
        vocab,
        log_size=args.log_size,
        use_generative_language=False,
        summary_writer=writer)
    validate(
        device,
        epoch,
        validation_data_loader,
        model,
        lstm_model,
        criterion,
        vocab,
        log_size=args.log_size,
        use_generative_language=False,
        summary_writer=writer)

  torch.save(model.state_dict(), args.model_save_dir)
  print('Trained model saved at ', args.model_save_dir)

  if args.summary_writer:
    writer.flush()
    writer.close()


if __name__ == '__main__':
  main()
