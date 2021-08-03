# Ask-Your-Humans
This repository is for the CS 7643 final project. We aim to reproduce the results achieved in the paper [*Ask Your Humans: Using Human Instructions to Improve Generalization in Reinforcement Learning*](https://arxiv.org/abs/2011.00517) in proceedings of ICLR 2021.

## Methods
+ Instruction Generator (iG) Pre-training
+ Imitation Learning (IL)
+ Imitation Learning with the assistance of Instruction Generator (IL-IG)

## Directory
+ *main.ipynb* - the main notebook
+ *config.py* - configuration for model structure, training process, etc.
+ *res* - resource including checkpoints, dataset, image, and image that we used in the paper
+ *src* - source code
+ *src/gamer* - gammer to interactivate with game environment
+ *src/mazebasev2* - game environment
+ *src/models* - models
+ *src/trainer* - trainer to train and validate models
+ *src/utils* - some helper functions
```Bash
Ask-Your-Humans
├── README.md
├── config.py
├── requirements.txt
├── main.ipynb
├── res
│   ├── cpts
│   ├── data
│   └── img
│       └── in_paper
└── src
    ├── gamer
    ├── mazebasev2
    ├── models
    ├── trainer
    └── utils
```

## Dependencies
+ python >= 3.8.5
+ tqdm >= 4.61.2
+ torch >= 1.7.1

## Setup
```Bash
conda create -n cs7643 python=3.8.5 pip
conda activate cs7643
git clone https://github.com/MrShininnnnn/Ask-Your-Humans.git
cd Ask-Your-Humans
pip install -r requirements.txt
# pip install numpy torch torchtext tensorboard matplotlib PyYAML jupyterlab
jupyter lab
```

## Note
+ It takes time to download Glove vector cache for the first run
+ Enable number of workers > 0 may cause allocate memory issue

## Authors
* **Ning Shi** - nshi30@gatech.edu
* **XingNan Zhou** - xzhou388@gatech.edu
* **Lei Lu** - llu79@gatech.edu
* **XingPeng Li** - xli409@gatech.edu