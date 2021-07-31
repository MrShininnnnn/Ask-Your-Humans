from args_loader import load_args
from dataloader import collate_fn
from dataloader import CraftingDataset
from dataloader import generate_vocab
from dataloader import load_pkl
from instructions_generator_metrics import InstructionsGeneratorMetrics
from instructions_generator_model import InstructionsGeneratorModel
import numpy as np
import random
from state_encoder import StateEncoder
from instructions_decoder import InstructionsDecoder
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


def train(grid_embedding, grid_onehot, inventory_embedding, goal_embedding,
          instructions, instructions_lengths, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, vocab, device):
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  encoder_outputs = encoder(grid_embedding, grid_onehot, inventory_embedding,
                            goal_embedding)
  batch_size = encoder_outputs.size(0)

  loss = 0
  decoder_input = torch.LongTensor([[vocab.word2idx['<start>']]] *
                                   batch_size).to(device)
  # decoder_input = torch.tensor(
  #     [[vocab.word2idx['<start>']] for i in range(batch_size)], device=device)
  # predictions = torch.zeros(batch_size, max(instructions_lengths),
  #                           len(vocab)).to(device)
  decoder_hidden = None

  use_teacher_forcing = True if random.random() < 0.5 else False

  if use_teacher_forcing:
    for di in range(max(instructions_lengths)):
      decoder_output, decoder_hidden, alphas = decoder(decoder_input,
                                                       decoder_hidden,
                                                       encoder_outputs)
      loss += criterion(decoder_output, target_tensor[di])
      decoder_input = target_tensor[di]

  else:
    for di in range(max(instructions_lengths)):
      decoder_output, decoder_hidden, alphas = decoder(decoder_input,
                                                       decoder_hidden,
                                                       encoder_outputs)
      _, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()

      loss += criterion(decoder_output, target_tensor[di])
      if decoder_input.item() == vocab.word2idx['<end>']:
        break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.item() / target_length


def trainIters(
    device,
    epoch,
    train_data_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    # parameters,
    vocab,
    log_size=500,
    summary_writer=None):
  # metrics = InstructionsGeneratorMetrics(vocab, criterion)
  running_loss = []
  all_losses = []
  encoder.train()
  decoder.train()

  for i, data in enumerate(train_data_loader, 0):
    grid_onehot, grid_embedding, inventory_embedding, _, goal_embedding, instructions, instructions_lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)
    instructions = instructions.to(device)

    loss = train(grid_embedding, grid_onehot, inventory_embedding,
                 goal_embedding, instructions, instructions_lengths,
                 instructions, encoder, decoder, encoder_optimizer,
                 decoder_optimizer, criterion, vocab, device)

    running_loss.append(loss)

    if i % log_size == log_size - 1:
      print('[%d, %5d] train loss: %.3f' %
            (epoch + 1, idx + 1, np.array(running_loss).mean()))

      all_losses.append(np.array(running_loss).mean())
      running_loss = []

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/train',
                              np.array(all_losses).mean(), epoch + 1)


def validate(device,
             epoch,
             val_loader,
             model,
             criterion,
             vocab,
             log_size=500,
             summary_writer=None):
  metrics = InstructionsGeneratorMetrics(vocab, criterion)
  model.eval()

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

  encoder = StateEncoder(args.embeded_dim).to(device)
  decoder = InstructionsDecoder(device, len(vocab), args.embeded_dim,
                                vocab_weights).to(device)

  criterion = nn.CrossEntropyLoss()
  # parameters = filter(lambda p: p.requires_grad, model.parameters())
  encoder_optimizer = torch.optim.Adam(
      encoder.parameters(), lr=args.learning_rate)
  decoder_optimizer = torch.optim.Adam(
      decoder.parameters(), lr=args.learning_rate)

  writer = SummaryWriter() if args.summary_writer else None

  # model.load_state_dict(
  #     torch.load(args.pretrained_model_dir))

  for epoch in range(args.epochs):
    trainIters(
        device,
        epoch,
        train_data_loader,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        criterion,
        vocab,
        log_size=args.log_size,
        summary_writer=writer)
    # validate(
    #     device,
    #     epoch,
    #     validation_data_loader,
    #     model,
    #     criterion,
    #     vocab,
    #     log_size=args.log_size,
    #     summary_writer=writer)

  torch.save(decoder.state_dict(), args.model_save_dir)
  print('Trained model saved at ', args.model_save_dir)

  if args.summary_writer:
    writer.flush()
    writer.close()


if __name__ == '__main__':
  main()
