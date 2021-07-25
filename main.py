import torch
import apps.dataloader as loader

### Test data loader

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda')
else:
    print("using cpu")
    device = torch.device('cpu')

workdir = '/mnt/e/DesignData/DL/Data_and_Code/dataset_split/'

train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions = loader.load_pkl(workdir)

vocab, vocab_weights = loader.generate_vocab(all_instructions, device=device, embed_dim=300, workdir = workdir)


