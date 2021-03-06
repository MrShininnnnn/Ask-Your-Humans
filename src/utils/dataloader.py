#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


# bulit-in
import os
import pickle
from .build_vocab import build_vocabulary
# public
import torch
import torchtext.vocab as vocabtorch
from torch.utils.data.dataset import Dataset
import numpy as np


def load_pkl(workdir, name_tag = 'dataset_'):

    if not os.path.exists(workdir):
        print('Please change to the correct directory.')
        return

    # name = workdir + name_tag
    name = os.path.join(workdir, name_tag)

    with open(name + 'states', 'rb') as f:
        train_states = pickle.load(f)

    with open(name + 'inventories', 'rb') as f:
        train_inventories = pickle.load(f)

    with open(name + 'actions', 'rb') as f:
        train_actions = pickle.load(f)

    with open(name + 'goals', 'rb') as f:
        train_goals = pickle.load(f)

    with open(name + 'instructions', 'rb') as f:
        train_instructions = pickle.load(f)

    with open(name + 'all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    return train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions


def generate_vocab(all_instructions,
                   device,
                   embed_dim = 300,
                   workdir = '/mnt/e/DesignData/DL/Data_and_Code/dataset_split/',
                   name_tag = 'dataset_',
                   glove_model = '840B',
                   cache=None):

    name = workdir + name_tag

    vocab, vocab_weights = build_vocabulary(all_instructions,
                                            name,
                                            embed_dim,
                                            glove_model=glove_model,
                                            cache=cache)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    #comment back when we define
    vocab_weights = torch.Tensor(vocab_weights).to(device)

    return vocab, vocab_weights


class CraftingDataset(Dataset):
    '''
    The class is copied from the original github.
    '''
    def __init__(self,
                 embed_dim,
                 train_states,
                 train_inventories,
                 train_actions,
                 train_goals,
                 train_instructions,
                 vocab,
                 transform=None,
                 glove_model='840B',
                 cache=None):

        self.embed_dim = embed_dim

        self.vocab = vocab
        self.train_instructions = train_instructions

        self.train_states = train_states
        self.train_inventories = train_inventories
        self.train_actions = train_actions
        self.train_goals = train_goals
        self.glove = vocabtorch.GloVe(
            name=glove_model, dim=embed_dim, cache=cache)

        self.train_states_embedding = [self.get_grid_embedding(state) for state in self.train_states]
        print("embedding loaded")
        self.train_states_onehot = [self.one_hot_grid(state) for state in self.train_states]
        print("one hot loaded")
        self.train_actions_onehot = [self.one_hot_actions(action) for action in self.train_actions]
        print("actions loaded")
        self.train_goals_embedding = [self.get_goal_embedding(goal) for goal in self.train_goals]
        print("goals loaded")
        self.train_inventory_embedding = [self.get_inventory_embedding(inventory) for inventory in self.train_inventories]
        print("done loading dataset")


    # input: multi-word crafting item string
    # output: summed glove word embedding (50d)
    def get_summed_embedding(self, phrase):

        phrase = phrase.split(' ')
        phrase_vector = torch.from_numpy(np.zeros((self.embed_dim), dtype=np.float32))

        for p in phrase:
            try:
                phrase_vector += self.glove.vectors[self.glove.stoi[p.lower()]]

            # MAKE THIS ALL zeros?
            except:
                phrase_vector += self.glove.vectors[self.glove.stoi['unknown']]  #replace this later??

        return phrase_vector

    # input: batched mazebase grid
    # output:
    def get_grid_embedding(self, batch_grid):

        goal_embedding_array = np.zeros((5, 5, self.embed_dim), dtype=np.float32)

        for x in range(5):
            for y in range(5):

                for index, item in enumerate(batch_grid[x][y]):
                    if item == "ResourceFont" or item == "CraftingContainer" or item == "CraftingItem":
                        goal_embedding_array[x][y] = self.get_summed_embedding(batch_grid[x][y][index+1])

        return goal_embedding_array

    def get_goal_embedding(self, goal):

            #currently all crafts are 2 word phrases
            # goal in the format of "Make Diamond Boots (Diamond Boots=1)" --> just extract diamond boots part

            goal_embedding = np.zeros((self.embed_dim), dtype=np.float32)

            goal = goal.split(' ')

            #item1_vec = self.glove.vectors[self.glove.stoi[goal[1].lower()]]
            #item2_vec = self.glove.vectors[self.glove.stoi[goal[2].lower()]]

            #goal_embedding = item1_vec+item2_vec

            goal_embedding = self.get_summed_embedding(goal[1]+' '+goal[2])

            return goal_embedding

    def get_inventory_embedding(self, inventory):


        #summed inventory
        inventory_embedding = np.zeros((self.embed_dim), dtype=np.float32)

        first = True
        for item in inventory:

            if inventory[item] > 0:

                if first:
                    inventory_embedding = self.get_summed_embedding(item)
                    first = False
                else:
                    inventory_embedding = inventory_embedding + self.get_summed_embedding(item)

        return inventory_embedding

    def one_hot_actions(self, action):

        if action == 'up':
            return np.array([1])
        elif action == 'down':
            return np.array([2])
        elif action == 'left':
            return np.array([3])
        elif action == 'right':
            return np.array([4])
        elif action == 'toggle_switch':
            return np.array([5])
        elif action == 'grab':
            return np.array([6])
        elif action == 'mine':
            return np.array([7])
        elif action == 'craft':
            return np.array([0])
        # stop is missing in the source code
        elif action == 'stop':
            return np.array([8])
        else:
            print(action)
            print('HEREEE')

    def one_hot_grid(self, grid):

        grid_embedding_array = np.zeros((5, 5, 7), dtype=np.float32)

        ## ADD information about switch and door opening!!

        for x in range(5):
            for y in range(5):

                for index, item in enumerate(grid[x][y]):

                    if item == 'Corner':
                        grid_embedding_array[x][y][0] = 1
                    elif item == 'Agent':
                        grid_embedding_array[x][y][1] = 1
                    elif item == 'Door' or item == 'Door_opened' or item == 'Door_closed':
                        grid_embedding_array[x][y][2] = 1
                    elif item == 'Key':
                        grid_embedding_array[x][y][3] = 1
                    elif item == 'Switch':
                        grid_embedding_array[x][y][4] = 1
                    elif item == 'Block':
                        grid_embedding_array[x][y][5] = 1
                    elif item == 'Door_closed':
                         grid_embedding_array[x][y][6] = 1

        return grid_embedding_array


    def __getitem__(self, index):

        states_embedding = torch.Tensor(self.train_states_embedding[index])
        states_onehot = torch.Tensor(self.train_states_onehot[index])
        action = torch.Tensor(self.train_actions_onehot[index])
        goal = torch.Tensor(self.train_goals_embedding[index])
        inventory = torch.Tensor(self.train_inventory_embedding[index])

        temp_instruction = self.train_instructions[index]

        # try:
        instruction = []
        instruction.append(self.vocab('<start>'))
        instruction.extend([self.vocab(token) for token in temp_instruction])
        instruction.append(self.vocab('<end>'))
        target = torch.Tensor(instruction)
        # except:
            #print(index)
            # instruction = [self.vocab('<unk>')]
            # target = torch.Tensor(instruction)

        #print(states_onehot.size(), states_embedding.size(), action.size(), goal.size())

        return states_onehot, states_embedding, inventory, action, goal, target

    def __len__(self):
        return len(self.train_states)
        # return self.train_states.shape[0]


def collate_fn(data):

    data.sort(key=lambda x: len(x[5]), reverse=True)
    states_onehot, states_embedding, inventory_embedding, action, goal, captions = zip(*data)

    states_onehot = torch.stack(states_onehot,0)
    states_embedding = torch.stack(states_embedding,0)
    action = torch.stack(action,0)
    goal = torch.stack(goal,0)
    inventory_embedding = torch.stack(inventory_embedding,0)


    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):

        end = lengths[i]
        targets[i, :end] = cap[:end]

    return states_onehot, states_embedding, inventory_embedding, action, goal, targets, lengths


def get_summed_embedding(phrase, glove, embed_size):

    phrase = phrase.split(' ')
    phrase_vector = torch.from_numpy(np.zeros((embed_size), dtype=np.float32))

    for p in phrase:
        phrase_vector += glove.vectors[glove.stoi[p.lower()]]

    return phrase_vector

def get_inventory_embedding(inventory, glove, embed_size):

    
    inventory_embedding = np.zeros((embed_size), dtype=np.float32)

    first = True
    for item in inventory:

        if inventory[item] > 0:

            if first:
                inventory_embedding = get_summed_embedding(item, glove, embed_size)
                first = False
            else:
                inventory_embedding = inventory_embedding + get_summed_embedding(item, glove, embed_size)

    return inventory_embedding

def get_grid_embedding(batch_grid, glove, embed_size):

    goal_embedding_array = np.zeros((5, 5, embed_size), dtype=np.float32)

    for x in range(5):
        for y in range(5):

            for index, item in enumerate(batch_grid[x][y]):
                if item == "ResourceFont" or item == "CraftingContainer" or item == "CraftingItem":
                    goal_embedding_array[x][y] = get_summed_embedding(batch_grid[x][y][index+1], glove, embed_size)
            
    return goal_embedding_array

def get_goal_embedding(goal, glove, embed_size):

    #currently all crafts are 2 word phrases
    # goal in the format of "Make Diamond Boots (Diamond Boots=1)" --> just extract diamond boots part

    goal_embedding = np.zeros((1,embed_size), dtype=np.float32)

    goal = goal.split(' ')

    item1_vec = glove.vectors[glove.stoi[goal[1].lower()]]
    item2_vec = glove.vectors[glove.stoi[goal[2].lower()]]

    goal_embedding[0] = item1_vec+item2_vec

    return goal_embedding

def one_hot_grid(grid, glove, embed_size):

    grid_embedding_array = np.zeros((5, 5, 7), dtype=np.float32)

    for x in range(5):
        for y in range(5):

            for index, item in enumerate(grid[x][y]):

                if item == 'Corner':
                    grid_embedding_array[x][y][0] = 1
                elif item == 'Agent':
                    grid_embedding_array[x][y][1] = 1
                elif 'Door' in item:
                    grid_embedding_array[x][y][2] = 1
                elif item == 'Key':
                    grid_embedding_array[x][y][3] = 1
                elif item == 'Switch':
                    grid_embedding_array[x][y][4] = 1
                elif item == 'Block':
                    grid_embedding_array[x][y][5] = 1
                elif item == 'closed': # door closed
                    grid_embedding_array[x][y][6] = 1

    return grid_embedding_array