from dataloader import load_pkl

bill_local_data_dir = "/usr/local/google/home/billzhou/Documents/dataset_split/"

train_states, train_inventories, train_actions, train_goals, train_instructions, all_instructions = load_pkl(
    workdir=bill_local_data_dir)

print(train_states.shape)
