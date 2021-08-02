#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# built-in
import os
# public
import torch


class Config(object):
    """docstring for Config"""
    def __init__(self):
        super().__init__()
        self.dev = False
        self.random_seed = 321
        self.summary_writer = True
        # see mazebasev2/options/knowledge_planner/
        self.game_file_name = 'length1task.yaml'
        # verify devices which can be either cpu or gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # I/O directory
        # current path
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        # resource path
        self.SOURCE_PATH = os.path.join(self.CURR_PATH, 'src')
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.CP_PATH = os.path.join(self.RESOURCE_PATH, 'cpts')
        if not os.path.exists(self.CP_PATH): os.makedirs(self.CP_PATH)
        self.GAME_ENV = os.path.join(self.SOURCE_PATH, 'mazebasev2')
        # '/usr/local/google/home/billzhou/Documents/glove'
        self.glove_cache = None
        # data settings
        self.grid_size = 5
        self.state_size = 25 # grid_size * grid_size
        self.state_onehot_size = 7
        self.action_size = 9


class InstrConfig(Config):
    """docstring for InstrConfig"""
    def __init__(self):
        super().__init__()
        # I/O directory
        self.SAVE_PATH = os.path.join(self.CP_PATH, 'instr_gen.pt')
        # train
        self.validation_split = 0.2
        self.num_workers = 0 if self.device == 'cuda' else 0
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.epochs = 20
        self.max_norm = 3
        self.teacher_forcing_rate = 1.0
        # valid
        self.valid_patience = 4
        # model
        self.embedding_dim = 300


class ILConfig(Config):
    """docstring for ILConfig"""
    def __init__(self):
        super().__init__()
        # I/O directory
        self.SAVE_PATH = os.path.join(self.CP_PATH, 'IL.pt')
        self.LSTM_SAVE_PATH = os.path.join(self.CP_PATH, 'LSTM.pt')
        # train
        self.validation_split = 0.2
        self.num_workers = 0 if self.device == 'cuda' else 0
        self.batch_size = 64
        self.learning_rate = 1e-3 # 0.001 in the paper but 0.003 in the source code
        self.epochs = 5
        self.max_norm = 3
        # valid
        self.valid_patience = 4
        # model
        self.embedding_dim = 300
        self.grid_embedding_hidden_size = 150
        self.grid_onehot_hidden_size = 20
        self.grid_hidden_size = 90
        self.goal_embedding_hidden_size = 150
        self.grid_goal_hidden_size = 512
        self.inventory_hidden_size = 50
        self.all_hidden_size = 48
        self.lstm_hidden_size = 32
