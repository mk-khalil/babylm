import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from load_pretrain_data import load_datasets_from_dir
import argparse
from utils import set_seed, Logger
import torch

class babyLMTokenizerTrainer:
    def __init__(self, model_path: str, vocab_size: int, model_max_length: int, special_tokens: list = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length)

        # Add special tokens if any
        if special_tokens is not None:
            self.tokenizer.add_special_tokens(special_tokens)

        self.vocab_size = vocab_size

    def __str__(self):
        return f"Tokenizer: {self.tokenizer}"

    def train_on_corpus(self, train_dataset, batch_size: int, full_corpus: bool = False, subset_length: int = None):
        """
        Trains the tokenizer on a given corpus, optionally using only a part of it.
        """
        if full_corpus is False:
            if subset_length is not None:
                dataset = self._select_dataset_subset(train_dataset, subset_length)
            else:
                raise ValueError("subset_length must be provided if full_corpus is True!")
        else:
            dataset = train_dataset
        return self._train_new_from_iterator(dataset, batch_size)

    def _select_dataset_subset(self, dataset, length: int):
        if dataset.num_rows < length:
            raise ValueError("length cannot be larger than the number of rows in the dataset!")
        idx = np.random.choice(np.arange(dataset.num_rows), length, replace=False)
        return dataset.select(idx)

    def _train_new_from_iterator(self, dataset, batch_size: int):
        def batch_iterator():
            for i in tqdm(range(0, len(dataset['text']), batch_size), desc="Training tokenizer from scratch on corpus"):
                yield dataset['text'][i:i + batch_size]
        self.tokenizer = self.tokenizer.train_new_from_iterator(batch_iterator(), vocab_size = self.vocab_size)
        return self.tokenizer

    def save(self, save_path: str):
        """ Saves the tokenizer to the specified path. """
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def load(save_path: str):
        return AutoTokenizer.from_pretrained(save_path)


# Main script for debugging
if __name__ == "__main__":

    # add parser argument for debug_or_train mode
    parser = argparse.ArgumentParser(description="Tokenizer Training Script")
    parser.add_argument("--mode", choices=['debug', 'train'], default='debug', help="Select the mode: debug or train.")
    parser.add_argument("--seed", type=int, default=69, help="Random seed.")

    args = parser.parse_args()
    debug = args.mode == 'debug'

    if debug:
        '''
        debug command usage: python tokenizer.py --mode debug
        '''
        print("Running in debug mode.")
        # ------------------------- Parameters:
        length = 1000  # num rows from dataset to process while training the tokenizer
        vocab_size = 12500  # newly trained tokenizer's target vocab size
        batch_size = 128  # batch size parameter of tokenizer.train_new_from_iterator() function
        tokenizer_model_or_path = "gpt2"  # 124M parameters, smallest version of GPT2
        tokenizer_save_path = "./save_dir/saved_tokenizer"
        model_max_length = 128  # Example max length for tokens
        # -------------------------------------

        # Initialize tokenizer trainer
        trainer = babyLMTokenizerTrainer(tokenizer_model_or_path, vocab_size, model_max_length)
        print(f"Base tokenizer loaded: {trainer}")

        # Load dataset (dummy function call here)
        train_dataset = load_datasets_from_dir(dataset_names=None, streaming=False)['train']
        if debug:
            train_dataset = train_dataset.select(range(0, 4000))  # reduce dataset size for debugging

        # Train the tokenizer on your corpus
        new_tokenizer = trainer.train_on_corpus(train_dataset, batch_size, full_corpus=False, subset_length=length)
        print(f"Tokenizer trained: {new_tokenizer}")

        # Save the newly trained tokenizer
        trainer.save(tokenizer_save_path)
        print(f"Tokenizer saved to {tokenizer_save_path}.")

        # Load the tokenizer back
        loaded_tokenizer = babyLMTokenizerTrainer.load(tokenizer_save_path)
        print(f"Loaded tokenizer: {loaded_tokenizer}")
    
    else:
        '''
        train command usage: python tokenizer.py --mode train --seed 420
        '''
        print("Running in train mode.")
        print(f'Using GPU: {torch.cuda.is_available()}')
        # ------------------------- Parameters:

        # parser parameters
        # Set seed
        set_seed(args.seed)

        # parameters
        tokenizer_save_path = "./save_dir/saved_tokenizer"
        tokenizer_model_or_path = "gpt2"  # tokenizer model used
        vocab_size = 16_000
        batch_size = 1024
        tokenizer_model_max_length = 128
        dataset_names = None
        full_corpus = True

        # Get reference to a Logger
        logger = Logger(log_file_name=tokenizer_save_path+'/logs/logger.logs', tb_log_dir=None)

        # load the whole training dataset
        print(f'Using dataset_names: {dataset_names if dataset_names is not None else "Loading all of the available .train, .dev, .test dataset files."}')

        train_dataset = load_datasets_from_dir(dataset_names=dataset_names, streaming=False, train_only= True)['train']

        # special tokens to be added to the tokenizer
        special_tokens = {  "bos_token": "<s>",
                            "eos_token": "</s>",
                            "unk_token": "<pad>"}
        # train the tokenizer
        tokenizer_trainer = babyLMTokenizerTrainer(tokenizer_model_or_path, vocab_size, tokenizer_model_max_length, special_tokens=special_tokens)

        trained_tokenizer = tokenizer_trainer.train_on_corpus(train_dataset, batch_size, full_corpus= full_corpus)

        # save the trained tokenizer
        tokenizer_trainer.save(tokenizer_save_path)


        logger.log_msg_to_console(f'Successfully trained the tokenizer: {trained_tokenizer}, and saved the tokenizer to path: {tokenizer_save_path}')

        params = {
            'tokenizer_save_path': tokenizer_save_path,
            'tokenizer_model_or_path': tokenizer_model_or_path,
            'vocab_size': vocab_size,
            'batch_size': batch_size,
            'tokenizer_model_max_length': tokenizer_model_max_length,
            'dataset_names': dataset_names
        }

        logger.log_dict_to_file(params) # log the parameters into a file