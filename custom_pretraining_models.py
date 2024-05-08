from os.path import join
import torch
from transformers import (
    AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig,
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
)
from transformers.modeling_outputs import TokenClassifierOutput

from load_pretrain_data import get_dataloaders

class customPretrainedConfig(PretrainedConfig):
    '''
    Config class for custom pretrained model
    '''
    model_type = "CustomPreTrainingTransformer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretraining_task = kwargs.get('pretraining_task', 'clm')  # default to 'clm'
        self.num_labels = kwargs.get('num_labels')  # should be set, raise error if not

class customPretrainedModel(PreTrainedModel):
    '''
    PreTrainedModel class for custom pretrained model
    '''
    config_class = customPretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        if config.pretraining_task not in ['clm', 'mlm']:
            raise ValueError("pretraining_task must be 'clm' or 'mlm'")

        self.encoder = AutoModel.from_config(config)
        self.head = self._get_head(config)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _get_head(self, config):
        if config.pretraining_task == 'clm':
            return AutoModelForCausalLM.from_config(config).lm_head
        # apply classification head for mlm task
        elif config.pretraining_task == 'mlm':
            return AutoModelForMaskedLM.from_config(config).cls

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        labels, input_ids, attention_mask = self._prepare_inputs_for_task(input_ids, attention_mask, labels)
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=False, **kwargs)
        logits = self.head(outputs['last_hidden_state'])

        loss = self._calculate_loss(logits, labels) if labels is not None else None
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def _prepare_inputs_for_task(self, input_ids, attention_mask, labels):

        if self.config.pretraining_task == 'clm' and labels is not None:
            labels = labels[..., 1:].contiguous()
            input_ids = input_ids[..., :-1].contiguous()
            attention_mask = attention_mask[..., :-1].contiguous()
        return labels, input_ids, attention_mask

    def _calculate_loss(self, logits, labels):
        #print(f"logits shape: {logits.shape}, labels shape: {labels.shape}")  # Add this line
        num_labels = self.config.num_labels
        #print(f"logits shape: {logits.reshape(-1, num_labels).shape}, labels shape {labels.reshape(-1).shape}")
        return self.loss_fn(logits.reshape(-1, num_labels), labels.reshape(-1))

    def save_model_and_config(self, model_save_path):
        self.config.save_pretrained(model_save_path)
        torch.save(self.state_dict(), join(model_save_path, "model_state_dict"))

    @staticmethod
    def load_saved_model(model_save_path):
        config = AutoConfig.from_pretrained(model_save_path)
        model = customPretrainedModel(config)
        model.load_state_dict(torch.load(join(model_save_path, "model_state_dict")))
        return model


#--------------------------debug--------------------------

def initialize_model(tokenizer, task):
    """ Initialize and return the custom model based on the tokenizer and task """
    config = AutoConfig.from_pretrained(tokenizer.name_or_path)
    config.num_labels = tokenizer.vocab_size
    config.vocab_size = tokenizer.vocab_size
    config.pretraining_task = task
    return customPretrainedModel(config)

def perform_forward_pass(model, dataloader):
    """ Perform a forward pass using the model and print the head of the model """
    for inputs in dataloader:
        output = model(**inputs)
        print(f"Output from the model: {output.loss}")
        break  # Only perform one forward pass for debugging
    print(f"Model head used for the task: {model.head}")

def save_and_load_model(model, path):
    """ Save and reload the model to ensure consistency """
    model.save_model_and_config(path)
    loaded_model = customPretrainedModel.load_saved_model(path)
    print(f"Loaded model head: {loaded_model}")
    # Ensure that the loaded model has the same parameters
    for original_param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(original_param, loaded_param), "Model parameters differ after loading."
    print("Model successfully saved and loaded with identical parameters.")

if __name__ == "__main__":
    model_name = 'gpt2'  # Chosen based on task index
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pretraining_task = 'clm'  # Chosen based on task index

    # Initialize Model from the config for the specified pretraining_task
    model = initialize_model(tokenizer, pretraining_task)
    #print(model.config)
    print(model.head)

    # Load a sample dataloader for testing (This function needs to be properly defined or imported)
    train_dataset_names = {
        'train': ['aochildes.train'],
        'validation': ['aochildes.dev'],
        'test': ['aochildes.test']
    }
    train_dataloader, _, _ = get_dataloaders(train_dataset_names, tokenizer, task=pretraining_task, batch_size=16, num_workers=0, return_small_debug_dataset=True)

    # Test model's forward pass
    perform_forward_pass(model, train_dataloader)

    # Test saving and loading custom model
    model_save_path = './save_dir/saved_customPretrainedModel'
    save_and_load_model(model, model_save_path)
