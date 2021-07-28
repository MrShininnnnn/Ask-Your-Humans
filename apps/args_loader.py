import argparse
import yaml


def load_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='./config.yaml')

  args = parser.parse_args()
  with open(args.config) as f:
    config = yaml.load(f)
  for key in config:
    for k, v in config[key].items():
      setattr(args, k, v)

  return args
