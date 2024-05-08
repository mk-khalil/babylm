import os
from custom_pretraining_models import customPretrainedModel
from tokenizer import babyLMTokenizerTrainer

'''
This script is used to load and save pretrained models to 
evaluate their performance using babylm evaluation pipeline 
available at https://github.com/babylm/evaluation-pipeline-2023
'''

# initiliaze the directory
baby_dir = os.getcwd()



# load the tokenizer once
tokenizer_path = os.path.join(baby_dir, 'save_dir/saved_tokenizer')
tokenizer = babyLMTokenizerTrainer.load(tokenizer_path)
print(f"Loaded tokenizer")


modes_paths = {
    'gpt2_cl': os.path.join(baby_dir, 'save_dir/gpt2_model1/training_loop_ckpt/harder_datasets/best_model_ckpt'),
    'bert_cl': os.path.join(baby_dir, 'save_dir/bert_model1/training_loop_ckpt/harder_datasets/best_model_ckpt'),
    'gpt2_base': os.path.join(baby_dir, 'save_dir/gpt2_model_base/training_loop_ckpt/full_datasets/best_model_ckpt'),
    'bert_base': os.path.join(baby_dir, 'save_dir/bert_model_base/training_loop_ckpt/full_datasets/best_model_ckpt')
}

# make sure all paths exist
for mode, path in modes_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f'{mode} path not found: {path}')

for i in modes_paths:

    print(f"Loading {i} model")
    model = customPretrainedModel.load_saved_model(modes_paths[i], )
    model.save_pretrained(os.path.join(baby_dir, f'models_eval/{i}'), safe_serialization=False)
    tokenizer.save_pretrained(os.path.join(baby_dir, f'models_eval/{i}'))
    print(f"{i} model saved")

'''
babylm evaluation pipeline commands
gpt_cl: python babylm_eval.py /home/ma2323/babylm/models_eval/gpt2_cl decoder
bert_cl: python babylm_eval.py /home/ma2323/babylm/models_eval/bert_cl encoder
gpt_base: python babylm_eval.py /home/ma2323/babylm/models_eval/gpt2_base decoder
bert_base: python babylm_eval.py /home/ma2323/babylm/models_eval/bert_base encoder

scp -r ma2323@34.74.129.118:/home/ma2323/babylm/models_eval/gpt2_cl /Users/ma2323/Downloads/modelsdir
scp -r ma2323@34.74.129.118:/home/ma2323/babylm/models_eval/bert_cl /Users/ma2323/Downloads/modelsdir
scp -r ma2323@34.74.129.118:/home/ma2323/babylm/models_eval/gpt2_base /Users/ma2323/Downloads/modelsdir
scp -r ma2323@34.74.129.118:/home/ma2323/babylm/models_eval/bert_base /Users/ma2323/Downloads/modelsdir

'''   