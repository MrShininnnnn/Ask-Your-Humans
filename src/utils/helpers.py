#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


# built-in
import random
# public
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_action_name(action):

    if action == 1:
        return 'up'
    elif action == 2:
        return 'down'
    elif action == 3:
        return 'left'
    elif action == 4:
        return 'right'
    elif action == 5:
        return  'toggle_switch'
    elif action == 6:
        return 'grab'
    elif action == 7:
        return 'mine'
    elif action  == 0:
        return 'craft'
    elif action == 8:
        return 'stop'