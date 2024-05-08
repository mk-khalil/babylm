import os
from datasets import load_dataset
from typing import Dict, Optional, Tuple
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import DataLoader


# ------------ Load train, validation, test datasets:

def get_data_files(data_dir, split):
  # returns list of file paths matching split (train,valid,test)
  return [data_dir + path for path in os.listdir(data_dir) if split in path]


def load_datasets_from_dir(dataset_names: Dict = None , streaming=False, train_only: bool = False):
    """
    Loads train, dev, test data for babylm strict small dataset and returns the corresponding huggingface DatasetDict
    If dataset_names is None, then all of the available train, dev, test sets are read.
    Returns:
        dataset (DatasetDict)
    """

    train_data_dir = 'babylm_data/babylm_10M_clean/'
    dev_data_dir = 'babylm_data/babylm_dev_clean/'
    test_data_dir = 'babylm_data/babylm_test/'

    if dataset_names is None: # load in all the available data files
        train_data_file_paths = get_data_files(train_data_dir, '.train')
        dev_data_file_paths = get_data_files(dev_data_dir, '.dev')
        test_data_file_paths = get_data_files(test_data_dir, '.test')
        data_files =  {
            'train': train_data_file_paths,
            'validation': dev_data_file_paths,
            'test': test_data_file_paths,
        }

    else: # load in specified data files
        data_files = {}
        for split_name, file_names in dataset_names.items():
            if split_name == 'train':
                file_names = list(map(lambda x: train_data_dir + x, file_names))
            if split_name == 'validation':
                file_names = list(map(lambda x: dev_data_dir + x, file_names))
            if split_name == 'test':
                file_names = list(map(lambda x: test_data_dir + x, file_names))

            data_files[split_name] = file_names
    if train_only:
        data_files = { 'train': data_files['train'] }

    dataset = load_dataset("text", data_files=data_files, streaming=streaming)
    return dataset

# Refer to: https://huggingface.co/learn/nlp-course/chapter7/6#preparing-the-dataset
def tokenize_dataset(raw_dataset, tokenizer):
    """
    Tokenizes the given raw_dataset(huggingface dataset with streaming=False) using the tokenizer and returns the tokenized dataset.
    No padding is done. Inputs are truncated if they are longer than the tokenizer.model_max_length.
    Returned rows are in 'input_ids' and their lengths are <= tokenizer.model_max_length.
    """
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            # not sure if context length should be used instead
            if length > 3:
                input_batch.append(input_ids)
        if len(input_batch) > 0:
            return {"input_ids": input_batch}
    #print("Tokenizing...")
    tokenized_dataset = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset["train"].column_names
    )
    return tokenized_dataset



# ------------ Process the text data (Pad, Mask, add labels) for Causal Language Modeling and Masked Language Modeling pretraining Tasks:

# Refer to: https://huggingface.co/learn/nlp-course/chapter7/6#preparing-the-dataset

def get_data_collator(task: str, tokenizer, mlm_probability: float = 0.15):
    """
    Returns a data collator for specified language modeling tasks. Supports 'clm' (causal language modeling)
    and 'mlm' (masked language modeling). The collator pads sequences to the length of the longest sequence,
    and adds label and attention_mask columns based on the task. Note that input and label alignment adjustments
    are handled within the model itself.

    Parameters:
        task (str): Type of the task ('clm' or 'mlm').
        tokenizer: Tokenizer instance for token manipulation.
        mlm_probability (float): Probability of masking for masked language modeling.

    Returns:
        DataCollatorForLanguageModeling: Configured data collator for the specified task.
    """
    assert task in ['clm', 'mlm'], "Task should be either 'clm' or 'mlm'."

    # Default assignment for pad_token and mask_token if they are not set
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None and tokenizer.eos_token:
        tokenizer.mask_token = tokenizer.eos_token

    # Define collator properties based on task
    is_mlm = task == 'mlm'
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=is_mlm,
        mlm_probability=mlm_probability if is_mlm else None,
        return_tensors='pt'
    )

    return collator

def get_dataloaders(  dataset_names: Optional[Dict[str, list]],
                      tokenizer: PreTrainedTokenizer,
                      task: str = 'clm',
                      batch_size: int = 32,
                      num_workers: int = 0,
                      return_small_debug_dataset: bool = False
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns three DataLoader instances for training, validation, and testing. It handles dataset loading,
    tokenization, and collation for the specified pretraining task using provided tokenizer.
    """
    assert task in ['clm', 'mlm'], "Task must be either 'clm' or 'mlm'."

    # Load and tokenize the dataset
    raw_dataset = load_datasets_from_dir(dataset_names=dataset_names, streaming=False)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)

    # Reduce dataset size if debugging
    if return_small_debug_dataset:
        #debug_slice = slice(40)  # Reduce the size to the first 40 samples for debugging
        debug_slice = list(range(10000))
        tokenized_dataset = {split: data.select(debug_slice) for split, data in tokenized_dataset.items()}

    # Prepare the data collator for the specified task
    data_collator = get_data_collator(task, tokenizer)

    # Create DataLoaders for each dataset split
    dataloaders = {
        split: DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=split == 'train',  # Only shuffle the training data
            num_workers=num_workers,
            collate_fn=data_collator
        )
        for split, data in tokenized_dataset.items()
    }

    return dataloaders['train'], dataloaders['validation'], dataloaders['test']


#----------------------debugging----------------------
def load_and_print_dataset(dataset_names):
    print(f'checking data loading:')
    raw_dataset = load_datasets_from_dir(dataset_names=dataset_names, streaming=False)
    print(raw_dataset)
    print('-' * 50)
    print(raw_dataset['train'])
    print('-' * 50)
    print(raw_dataset['train']['text'][:5])
    print('-' * 100)
    return raw_dataset

def tokenize_and_print_dataset(raw_dataset, tokenizer):
    print(f'checking tokenization:')
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    print(tokenized_dataset)
    print('-' * 50)
    print(tokenized_dataset['train']['input_ids'][:5])
    print('-' * 100)
    return tokenized_dataset

def check_data_collators(tokenized_dataset, tokenizer):
    for task in ['clm', 'mlm']:
        print(f'checking {task} data collator:')
        collator = get_data_collator(task, tokenizer)
        example_batch = [tokenized_dataset["train"][i] for i in range(5)]
        output = collator(example_batch)
        print(output)
        print('-' * 100)


if __name__ == "__main__":
    train_dataset_names = {
        'train': ['aochildes.train'],
        'validation': ['aochildes.dev'],
        'test': ['aochildes.test']
    }
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    raw_dataset = load_and_print_dataset(train_dataset_names)
    tokenized_dataset = tokenize_and_print_dataset(raw_dataset, tokenizer)
    check_data_collators(tokenized_dataset, tokenizer)