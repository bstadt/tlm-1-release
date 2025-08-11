import os
import wandb
import click
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from loader import DataCollatorForTemporalSpanMasking, JSONLTokenizedDataset
from transformers import BertTokenizerFast, TrainingArguments, Trainer, BertConfig, BertForMaskedLM, AutoTokenizer, ModernBertForMaskedLM, AutoModelForMaskedLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
import torch
import torch.nn.functional as F

os.environ["WANDB_PROJECT"] = "tlm"

class TLMMetricsCallback(TrainerCallback):
    def __init__(self):
        self.current_batch_first_token_accuracy = 0.0
        self.current_batch_infill_accuracy = 0.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs['first_token_accuracy'] = self.current_batch_first_token_accuracy
            logs['infill_accuracy'] = self.current_batch_infill_accuracy
    
    def update_metrics(self, logits, labels):
        """Update the batch-wise accuracy with new predictions"""
        # Get the first token predictions (position 0)
        first_token_logits = logits[:, 0, :]  # Shape: (batch_size, vocab_size)
        first_token_labels = labels[:, 0]  # Shape: (batch_size,)

        # Calculate accuracy for first token predictions
        # Only consider tokens that are actually masked (not -100)
        mask_indices = first_token_labels != -100
        if mask_indices.sum() > 0:
            first_token_preds = torch.argmax(first_token_logits[mask_indices], dim=-1)
            first_token_correct = (first_token_preds == first_token_labels[mask_indices]).sum().item()
            first_token_total = mask_indices.sum().item()
            self.current_batch_first_token_accuracy = first_token_correct / first_token_total
        else:
            self.current_batch_first_token_accuracy = 0.0

        # Compute accuracy for all other mask fills except the first token
        # Exclude position 0
        other_token_labels = labels[:, 1:]  # Shape: (batch_size, seq_len-1)
        other_token_logits = logits[:, 1:, :]  # Shape: (batch_size, seq_len-1, vocab_size)
        # Flatten batch and sequence dims for easier computation
        other_mask_indices = other_token_labels != -100  # (batch_size, seq_len-1)
        if other_mask_indices.sum() > 0:
            # Get predictions for masked positions
            preds = torch.argmax(other_token_logits, dim=-1)  # (batch_size, seq_len-1)
            correct = (preds == other_token_labels) & other_mask_indices
            other_token_correct = correct.sum().item()
            other_token_total = other_mask_indices.sum().item()
            self.current_batch_infill_accuracy = other_token_correct / other_token_total
        else:
            self.current_batch_infill_accuracy = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs['first_token_accuracy'] = self.current_batch_first_token_accuracy
            logs['infill_accuracy'] = self.current_batch_infill_accuracy
            wandb.log(logs)

class TLMTrainer(Trainer):
    def __init__(self, *args, temporal_loss_weight=1./512, grad_accum_correction=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tlm_metrics_callback = TLMMetricsCallback()
        self.temporal_loss_weight = temporal_loss_weight
        self.grad_accum_correction = grad_accum_correction
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Update first token accuracy
        if 'labels' in inputs:
            self.tlm_metrics_callback.update_metrics(logits, inputs['labels'])

        # Custom loss: multiply first token loss by a hyperparameter
        first_token_loss_weight = self.temporal_loss_weight
        labels = inputs['labels']
        # Flatten logits and labels for cross-entropy
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        # Compute loss for all tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        all_losses = loss_fct(logits_flat, labels_flat).view_as(labels)
        # First token loss (position 0)
        first_token_loss = all_losses[:, 0]
        # Other tokens loss (positions 1+)
        other_token_loss = all_losses[:, 1:]
        # Mask for valid tokens
        first_token_mask = (labels[:, 0] != -100)
        other_token_mask = (labels[:, 1:] != -100)
        # Weighted sum
        first_token_loss_sum = (first_token_loss * first_token_mask).sum()
        first_token_count = first_token_mask.sum()
        other_token_loss_sum = (other_token_loss * other_token_mask).sum()
        other_token_count = other_token_mask.sum()
        # Avoid division by zero
        first_token_loss_mean = first_token_loss_sum / (first_token_count + 1e-8)
        other_token_loss_mean = other_token_loss_sum / (other_token_count + 1e-8)
        loss = first_token_loss_weight * first_token_loss_mean + other_token_loss_mean
        if self.grad_accum_correction:
            loss = loss / self.args.gradient_accumulation_steps
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def log(self, logs, *args, **kwargs):
        # Add first token accuracy to logs
        logs['first_token_accuracy'] = self.tlm_metrics_callback.current_batch_first_token_accuracy
        logs['infill_accuracy'] = self.tlm_metrics_callback.current_batch_infill_accuracy
        super().log(logs, *args, **kwargs)

@click.command()
@click.option('--dataset', type=click.Choice(['coca', 'now']), required=True, help='Dataset to use for training')
def main(dataset):
    

    print('Loading tokenizer...')
    if dataset == 'coca':
        tokenizer_path = './coca_tokenized_ettin_large/tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:  # now

        tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-400m")

        # Add new special tokens
        additional_special_tokens = ['[MASK_NOLOSS]'] 
        for year in range(10, 26):
            for month in range(1, 13):
                additional_special_tokens.append('[TIME:{i}-{j}]'.format(i=year, j=month))
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)

        # save the tokenizer
        os.makedirs('./now_tokenized_ettin_large/tokenizer/', exist_ok=True)
        tokenizer.save_pretrained('./coca_tokenized_ettin_large/tokenizer/')


    print('Loading training config...')
    run_name='{}-tlm-{}'.format(dataset, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    training_args = TrainingArguments(
        output_dir='./{}'.format(run_name),
        num_train_epochs=2,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=64,
        warmup_steps=5000,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_dir='./logs/{}'.format(run_name),
        logging_steps=10,
        bf16=True,
        report_to='wandb',
        run_name=run_name,
    )

    print('Loading model...')
    if dataset == 'now':
        print('using random init')
        import yaml
        with open('./encoder_large_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        model_config = config['model']['model_config']
        model_config = BertConfig(**model_config)
        model = BertForMaskedLM._from_config(model_config)
    else:
        model = AutoModelForMaskedLM.from_pretrained('jhu-clsp/ettin-encoder-400m', trust_remote_code=True)


    if dataset == 'coca':
        # For COAA we are fine tuning
        # Set new special token embeddings to the embedding of the space character
        # This is a hack to get the model to work with the new special tokens
        # Otherwise, all of the model infra freaks out and we get mega loss
        additional_special_tokens = ['[MASK_NOLOSS]'] + ['[YEAR:{i}]'.format(i=i) for i in range(1990, 2025)]
        space_token_id = tokenizer.convert_tokens_to_ids(' ')
        if space_token_id == tokenizer.unk_token_id:
            space_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        embedding = model.get_input_embeddings()
        new_token_ids = tokenizer.convert_tokens_to_ids(additional_special_tokens)
        space_embedding = embedding.weight.data[space_token_id].clone()
        for token_id in new_token_ids:
            embedding.weight.data[token_id] = space_embedding

    model.resize_token_embeddings(len(tokenizer))

    print(f'Loading {dataset} dataset...')
    if dataset == 'coca':
        dataset_path = './coca_tokenized_ettin_large'
    else:  # now
        dataset_path = './now_tokenized_ettin_large'
    
    dataset = JSONLTokenizedDataset(root_path=dataset_path, debug=False)

    data_collator = DataCollatorForTemporalSpanMasking(tokenizer, num_spans=55)

    print('Loading trainer...')
    trainer = TLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        temporal_loss_weight=0.5,
        grad_accum_correction=dataset=='coca'
    )

    print('Training!')
    trainer.train()

    print('Saving model...')
    trainer.save_model('./models/{}'.format(run_name))

if __name__ == "__main__":
    main()