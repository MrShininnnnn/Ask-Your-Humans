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
        self.dev = True
        self.random_seed = 123
        self.summary_writer = True
        # verify devices which can be either cpu or gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # I/O directory
        # current path
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        # resource path
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.CP_PATH = os.path.join(self.RESOURCE_PATH, 'cpts')
        if not os.path.exists(self.CP_PATH): os.makedirs(self.CP_PATH)

class InstrConfig(Config):
    """docstring for InstrConfig"""
    def __init__(self):
        super(InstrConfig, self).__init__()
        # I/O directory
        self.SAVE_PATH = os.path.join(self.CP_PATH, 'instr_gen.pt')
        # train
        self.validation_split = 0.2
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.epochs = 20
        # valid
        self.valid_patience = 4
        # model
        self.embeded_dim=300
        