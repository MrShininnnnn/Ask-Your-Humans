#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


# built-in
import os
import yaml
import json
# private
from ..mazebasev2.lib.mazebase import games
from ..mazebasev2.lib.mazebase.games import featurizers


# adapted from "https://github.com/valeriechen/ask-your-humans/blob/d9b5a5dc2e29c369ed9582b4473720f928e92f50/mazebase-training/test_models.py"
def generate_new_game(config):
    yaml_file = os.path.join(config.GAME_ENV, 'options', 'knowledge_planner', config.game_file_name)
    with open(yaml_file, 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)
    # Get sub opts
    method_opt = options['method']
    env_opt = options['env']
    log_opt = options['logs'] 
    # Set up the mazebase environment
    knowledge_root = env_opt['knowledge_root']
    world_knowledge_file = os.path.join(config.GAME_ENV, knowledge_root, env_opt['world_knowledge']['train'])
    with open(world_knowledge_file) as f:
        world_knowledge = json.load(f)

    # Make the world
    map_size = (env_opt['state_rep']['w'], env_opt['state_rep']['w'], env_opt['state_rep']['h'], env_opt['state_rep']['h'])
    all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=[], options=env_opt, load_items=None, map_size=map_size)]

    # Game wrapper
    game = games.MazeGame(
      all_games,
      featurizer=featurizers.GridFeaturizer()
    )

    return game