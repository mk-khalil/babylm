from utils import set_seed, Logger
import argparse
from tokenizer import babyLMTokenizerTrainer
from transformers import AutoConfig
from custom_pretraining_models import customPretrainedModel
from hf_training_loop import train_model_hf

# Set up argument parser
parser = argparse.ArgumentParser(description="Train GPT-2 or BERT models.")
parser.add_argument("--model_type", choices=["gpt2", "bert"], required=True, help="Type of model to train (gpt2 or bert).")

#parser.add_argument("--training_mode", choices=['base', 'CL'], required=True, help="Type of training mode (base or cirriculum learning).")

# Split training dataset into easy and hard subsets based on manual assessment
easier_dataset_names = ['aochildes.train', 'open_subtitles.train', 'qed.train', 'simple_wikipedia.train', 'switchboard.train']
harder_dataset_names = ['bnc_spoken.train', 'cbt.train', 'children_stories.train', 'gutenberg.train', 'wikipedia.train']

all_train_dataset_names = ['aochildes.train', 'gutenberg.train', 'switchboard.train', 'bnc_spoken.train', 'open_subtitles.train', 'wikipedia.train', 'cbt.train', 'qed.train', 'children_stories.train', 'simple_wikipedia.train']
all_val_dataset_names = ['aochildes.dev', 'gutenberg.dev', 'switchboard.dev', 'bnc_spoken.dev', 'open_subtitles.dev', 'wikipedia.dev', 'cbt.dev', 'qed.dev', 'children_stories.dev', 'simple_wikipedia.dev']
all_test_dataset_names = ['aochildes.test', 'gutenberg.test', 'switchboard.test', 'bnc_spoken.test', 'open_subtitles.test', 'wikipedia.test', 'cbt.test', 'qed.test', 'children_stories.test', 'simple_wikipedia.test']


'''
In this script, we will do the following steps:
- load our trained tokenizer
- train a custom gpt2/bert model using the easier datasets
- save the model
- load the model and train it using the harder datasets
- save the model and test it
'''

# Parse arguments
args = parser.parse_args()


# Define parameters based on model type
if args.model_type == "gpt2":
    model_params = {
        'tokenizer_path': "./save_dir/saved_tokenizer",
        'transformer_model_name': 'gpt2',
        'pretraining_task': 'clm',
        'model_ckpt_easy': "./save_dir/gpt2_model1/training_loop_ckpt/easier_datasets",
        'model_ckpt_hard': "./save_dir/gpt2_model1/training_loop_ckpt/harder_datasets",
    }
else:  # bert
    model_params = {
        'tokenizer_path': "./save_dir/saved_tokenizer",
        'transformer_model_name': 'bert-base-uncased',
        'pretraining_task': 'mlm',
        'model_ckpt_easy': "./save_dir/bert_model1/training_loop_ckpt/easier_datasets",
        'model_ckpt_hard': "./save_dir/bert_model1/training_loop_ckpt/harder_datasets",
    }

# Common parameters for both models
common_params = {
    'batch_size': 32,
    'num_epochs': 3,
    'hidden_size': 256,
    'hidden_layers': 5 if args.model_type == "gpt2" else 4,
    'attention_heads': 8 if args.model_type == "gpt2" else 4,
    'lr': 1e-4
}

# Merge dictionaries
params = {**model_params, **common_params}

# Set seed
set_seed(420)

#----------------------------- easy training -----------------------------#

# initialize logger
logger = Logger(log_file_name=params['model_ckpt_easy'] + '/logs/logger.logs', tb_log_dir=params['model_ckpt_easy'] + '/logs/tb_logs')
logger.log_dict_to_file(params)

# define dataset names
dataset_names = { 
    'train': easier_dataset_names,
    'validation': all_val_dataset_names,
    'test': all_test_dataset_names
}

# Load the tokenizer once
tokenizer = babyLMTokenizerTrainer.load(params['tokenizer_path'])
logger.log_msg_to_console(f'Loaded tokenizer: {tokenizer}, from path: {params["tokenizer_path"]}')

# initialize the model
config = AutoConfig.from_pretrained(params['transformer_model_name'])
config.num_labels = tokenizer.vocab_size # set the number of labels to the vocab size of the tokenizer
config.vocab_size = tokenizer.vocab_size 
config.pretraining_task = params['pretraining_task']

if args.model_type == "gpt2":
    config.n_embd = params['hidden_size'] # defaults to 768
    config.n_head = params['attention_heads'] # defaults to 12
    config.n_layer = params['hidden_layers'] # defaults to 12

elif args.model_type == "bert":
    config.hidden_size = params['hidden_size'] # defaults to 768
    config.num_attention_heads = params['attention_heads'] # defaults to 12
    config.num_hidden_layers = params['hidden_layers'] # defaults to 12
else:
    raise ValueError("Invalid model type")

logger.log_to_file(f'Config: {config}')

model = customPretrainedModel(config)
logger.log_to_file(f'Initialized model: {model}, with size: {sum(t.numel() for t in model.parameters())/1000**2:.1f}M parameters')


# start training using the hugging face training loop
train_model_hf(dataset_names, model, tokenizer, params['pretraining_task'], params['batch_size'], params['num_epochs'], params['model_ckpt_easy'], lr=float(params['lr']))

logger.log_msg_to_console(f'Successfully trained the model on the easier datasets, saved the model to path: {params["model_ckpt_easy"]}')
logger.log_to_file(f'Successfully trained the model on the easier datasets, saved the model to path: {params["model_ckpt_easy"]}')

#----------------------------- hard training -----------------------------#

# Load the model from the previous training
model = customPretrainedModel.load_saved_model(params['model_ckpt_easy'] + "/best_model_ckpt")

model_size = sum(t.numel() for t in model.parameters())
logger.log_to_file(f'Successfully loaded model: {model} \nwith size: {model_size/1000**2:.1f}M parameters')

# Resume training on the harder dataset and test the model
test_results = train_model_hf(dataset_names, model, tokenizer, params['pretraining_task'], params['batch_size'], params['num_epochs'], params['model_ckpt_hard'], lr=float(params['lr']), test_model=True)

logger.log_msg_to_console(f'Successfully resumed training and testing the model, saved the model to path: {params["model_ckpt_hard"]}')
logger.log_to_file(f'Successfully resumed training and testing the model, saved the model to path: {params["model_ckpt_hard"]}')
logger.log_to_file(f'Test results: {test_results}')