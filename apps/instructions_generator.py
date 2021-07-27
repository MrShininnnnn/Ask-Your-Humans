import torch

from dataloader import load_pkl, generate_vocab, CraftingDataset

# TODO(xingnan): Load these parameters from config file
DATA_DIR = '/usr/local/google/home/billzhou/Documents/dataset_split/'
EMBEDED_DIM = 300
VECTORS_CACHE = '/usr/local/google/home/billzhou/Documents/glove'

if torch.cuda.is_available():
  print('using cuda')
  device = torch.device('cuda')
else:
  print('using cpu')
  device = torch.device('cpu')

train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions = load_pkl(
    workdir=DATA_DIR)

vocab, _ = generate_vocab(
    all_instructions,
    device,
    workdir=DATA_DIR,
    cache=VECTORS_CACHE)

dataset = CraftingDataset(
    EMBEDED_DIM,
    train_states,
    train_inventories,
    train_actions,
    train_goals,
    train_instructions,
    vocab,
    cache=VECTORS_CACHE)
