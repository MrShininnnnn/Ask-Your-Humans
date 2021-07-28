from args_loader import load_args
from dataloader import collate_fn
from dataloader import CraftingDataset
from dataloader import generate_vocab
from dataloader import load_pkl
from instructions_generator_metrics import InstructionsGeneratorMetrics
from instructions_generator_model import InstructionsGeneratorModel
import numpy as np
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


def calculate_bleu_score(preds, targets, vocab):
  preds_words = np.array([
      list(map(lambda x: vocab.idx2word[x], preds[i]))
      for i in range(preds.shape[0])
  ])
  targets_words = np.array(
      [[list(map(lambda x: vocab.idx2word[x], targets[i]))]
       for i in range(targets.shape[0])])
  return bleu_score(preds_words, targets_words)


def train(device,
          epoch,
          train_data_loader,
          model,
          optimizer,
          criterion,
          parameters,
          vocab,
          log_size=500,
          summary_writer=None):
  metrics = InstructionsGeneratorMetrics(vocab, criterion)

  for i, data in enumerate(train_data_loader, 0):
    grid_onehot, grid_embedding, inventory_embedding, _, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    instructions = instructions.to(device)

    predictions, decode_lengths, alphas, _ = model(grid_embedding, grid_onehot,
                                                   inventory_embedding,
                                                   goal_embedding, instructions,
                                                   lengths)

    targets = instructions[:, 1:]
    try:
      lang_loss = metrics.add(predictions, targets, decode_lengths, alphas)
      clip_grad_norm_(parameters, max_norm=3)

      optimizer.zero_grad()
      lang_loss.backward()
      optimizer.step()

    except RuntimeError as error:
      print(error)

    if i % log_size == log_size - 1:
      metrics.flush(epoch, i)

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/train',
                              metrics.get_mean(metrics.all_losses), epoch + 1)
    summary_writer.add_scalar('Bleu/train',
                              metrics.get_mean(metrics.all_bleu_scores),
                              epoch + 1)
    summary_writer.add_scalar('TokenAcc/train',
                              metrics.get_mean(metrics.all_token_accuracy),
                              epoch + 1)


def validate(device,
             epoch,
             val_loader,
             model,
             criterion,
             vocab,
             log_size=500,
             summary_writer=None):
  metrics = InstructionsGeneratorMetrics(vocab, criterion)

  for idx, data in enumerate(val_loader, 0):
    grid_onehot, grid_embedding, inventory_embedding, _, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    instructions = instructions.to(device)

    with torch.no_grad():
      predictions, decode_lengths, alphas, _ = model(grid_embedding,
                                                     grid_onehot,
                                                     inventory_embedding,
                                                     goal_embedding,
                                                     instructions, lengths)
      targets = instructions[:, 1:]
      try:
        metrics.add(predictions, targets, decode_lengths, alphas)

      except RuntimeError as error:
        print(error)

    if idx % log_size == log_size - 1:
      metrics.flush(epoch, idx, train=False)

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/valid',
                              metrics.get_mean(metrics.all_losses), epoch + 1)
    summary_writer.add_scalar('Bleu/valid',
                              metrics.get_mean(metrics.all_bleu_scores),
                              epoch + 1)
    summary_writer.add_scalar('TokenAcc/valid',
                              metrics.get_mean(metrics.all_token_accuracy),
                              epoch + 1)


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

  instructions_generator = InstructionsGeneratorModel(device, len(vocab),
                                                      args.embeded_dim,
                                                      vocab_weights)
  instructions_generator.to(device)
  instructions_generator.train()

  criterion = nn.CrossEntropyLoss()
  parameters = filter(lambda p: p.requires_grad,
                      instructions_generator.parameters())
  optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

  writer = SummaryWriter() if args.summary_writer else None

  for epoch in range(args.epochs):
    train(
        device,
        epoch,
        train_data_loader,
        instructions_generator,
        optimizer,
        criterion,
        parameters,
        vocab,
        log_size=args.log_size,
        summary_writer=writer)
    validate(
        device,
        epoch,
        validation_data_loader,
        instructions_generator,
        criterion,
        vocab,
        log_size=args.log_size,
        summary_writer=writer)

  torch.save(instructions_generator.state_dict(), args.model_save_dir)
  print('Trained model saved at ', args.model_save_dir)

  if args.summary_writer:
    writer.flush()
    writer.close()

  return instructions_generator


if __name__ == '__main__':
  main()
