from dataloader import collate_fn
from dataloader import CraftingDataset
from dataloader import generate_vocab
from dataloader import load_pkl
from instructions_generator_model import InstructionsGeneratorModel
import torch
from torch.utils.data import DataLoader

# TODO(xingnan): Load these parameters from config file
DATA_DIR = '/usr/local/google/home/billzhou/Documents/dataset_split/'
EMBEDED_DIM = 300
VECTORS_CACHE = '/usr/local/google/home/billzhou/Documents/glove'
BATCH_SIZE = 64
EPOCHS = 1  # Change to 15

if torch.cuda.is_available():
  print('using cuda')
  device = torch.device('cuda')
else:
  print('using cpu')
  device = torch.device('cpu')

train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions = load_pkl(
    workdir=DATA_DIR)

vocab, vocab_weights = generate_vocab(
    all_instructions, device, workdir=DATA_DIR, cache=VECTORS_CACHE)

dataset = CraftingDataset(
    EMBEDED_DIM,
    train_states,
    train_inventories,
    train_actions,
    train_goals,
    train_instructions,
    vocab,
    cache=VECTORS_CACHE)
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn)

instructions_generator = InstructionsGeneratorModel(
    len(vocab), EMBEDED_DIM, vocab_weights)
instructions_generator.to(device)

instructions_generator.train()

for epoch in range(EPOCHS):
  for i, data in enumerate(data_loader, 0):
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    encoder_out = instructions_generator(grid_embedding, grid_onehot,
                                         inventory_embedding, goal_embedding)
    print(encoder_out)
