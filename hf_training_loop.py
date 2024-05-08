from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
from load_pretrain_data import load_datasets_from_dir, tokenize_dataset, get_data_collator
import os
def setup_training_arguments(batch_size, num_epochs, log_dir, lr):
    """
    Create and return TrainingArguments for the Hugging Face Trainer.
    """
    return TrainingArguments(
        output_dir= log_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=10_000,
        logging_steps=1_000,
        logging_dir=log_dir,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=lr,
        save_safetensors = False,
        save_steps=10_000,
        load_best_model_at_end=True,
        fp16=True,
    )

# define metrics and metrics function
accuracy_metric = evaluate.load("accuracy")

def get_compute_metrics_fn(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": acc["accuracy"],
    }

def initialize_trainer(model, tokenizer, train_dataset, eval_dataset, collate_fn, compute_metrics_fn, args):
    """
    Initialize and return the Hugging Face Trainer.
    """
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        #compute_metrics= get_compute_metrics_fn
    )

# def get_compute_metrics_fn():
#     """
#     Define and return a compute_metrics function to evaluate model performance.
#     """
#     metric = evaluate.load("accuracy")

#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         predictions = np.argmax(logits, axis=-1)
#         return metric.compute(predictions=predictions, references=labels)

#     return compute_metrics



def train_model_hf(train_dataset_names, model, tokenizer, pretraining_task, batch_size, num_epochs, log_dir, lr=5e-4, test_model=False):
    """
    Train the model using the Hugging Face Trainer setup.
    """
    # get dataset dict
    raw_dataset = load_datasets_from_dir(dataset_names=train_dataset_names, streaming=False)
    # slice raw_dataset for debugging
    #raw_dataset = {split: data.select(range(10000)) for split, data in raw_dataset.items()}

    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)

    # print tokenized dataset
    print(tokenized_dataset)

    collate_fn = get_data_collator(pretraining_task, tokenizer)
    #compute_metrics_fn = get_compute_metrics_fn()
    compute_metrics_fn = None
    args = setup_training_arguments(batch_size, num_epochs, log_dir, lr)
    trainer = initialize_trainer(
        model, tokenizer, tokenized_dataset["train"], tokenized_dataset["validation"],
        collate_fn, compute_metrics_fn, args
    )

    trainer.train()


    # Save the trained model and its configuration.
    # handle distributed (parallel) training
    trained_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    trained_model.save_model_and_config(model_save_path=log_dir+"/best_model_ckpt")

    # Evaluate the model on the test set if test_model is True
    if test_model:
        eval_results = trainer.evaluate(tokenized_dataset["test"])
        print(f"Test set evaluation results: {eval_results}")
        return eval_results
    else:
        return None
