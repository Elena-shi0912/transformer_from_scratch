from dataset import EnFrDataset
from tokenizer import load_tokenizer
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset

def get_and_load_dataset(config):
    # load dataset and get the train split
    dataset_raw = load_dataset(f"{config['dataset']}", f"{config['source_language']}-{config['target_language']}", split='train')
    
    # build tokenizers
    tokenizer_src = load_tokenizer(config, dataset_raw, config["source_language"])
    tokenizer_target = load_tokenizer(config, dataset_raw, config["target_language"])
    
    # keep 90% for training, 10% for validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])
    
    train_dataset = EnFrDataset(train_dataset_raw, tokenizer_src, tokenizer_target, config["source_language"], config["target_language"], config["seq_len"])
    val_dataset = EnFrDataset(val_dataset_raw, tokenizer_src, tokenizer_target, config["source_language"], config["target_language"], config["seq_len"])
    
    # Find the maximum length of each sentence in the source and target sentence
    max_len_source = 0
    max_len_target = 0
    
    for item in dataset_raw:
        source_ids= tokenizer_src.encode(item["translation"][config["source_language"]]).ids
        target_ids = tokenizer_target.encode(item["translation"][config["target_language"]]).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))
        
    print(f"Max length of source sentence: {max_len_source}")
    print(f"Max length of target sentence: {max_len_target}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target
