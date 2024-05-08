# ENLP Course Project

This repository contains all the necessary code for the ENLP course project at Georgetown University. It includes scripts for training baseline and curriculum learning models on the BabyLM dataset.

## Instructions to run

To download and prepare the dataset:

1. Download the BabyLM dataset:
   ```bash
   wget https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip
   ```

2. Unzip the dataset in the project directory:
   ```bash
   unzip babylm_data.zip
   ```


Install the required dependencies:
``` bash 
pip install -r requirements.txt
```

Clean the dataset before training the models:
``` bash
python clean_data.py
```
### Training Models

To train the baseline models using the complete dataset:
``` bash
python train_base.py --model_type <gpt, bert>
```

Replace <gpt, bert> with 'gpt' or 'bert' depending on the model you want to train.

To train models using curriculum learning:
``` bash
python train.py --model_type <gpt, bert>
``` bash

Again, replace <gpt, bert> with 'gpt' or 'bert' as required.
