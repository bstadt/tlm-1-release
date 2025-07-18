import os
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from loader import DataCollatorForTemporalSpanMasking, COCATokenizedDataset
from transformers import BertTokenizerFast, TrainingArguments, Trainer, BertConfig, BertForMaskedLM
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #run_name = 'tlm-{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        #wandb.init(project='tlm', name=run_name, config=args)
        self.tlm_metrics_callback = TLMMetricsCallback()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Update first token accuracy
        if 'labels' in inputs:
            self.tlm_metrics_callback.update_metrics(logits, inputs['labels'])
        
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def log(self, logs, *args, **kwargs):
        # Add first token accuracy to logs
        logs['first_token_accuracy'] = self.tlm_metrics_callback.current_batch_first_token_accuracy
        logs['infill_accuracy'] = self.tlm_metrics_callback.current_batch_infill_accuracy
        super().log(logs, *args, **kwargs)

if __name__ == "__main__":
    
    print('Loading dataset...')
    dataset = COCATokenizedDataset(root_path='./coca_tokenized', debug=False)

    print('Loading tokenizer...')
    tokenizer = BertTokenizerFast.from_pretrained('./coca_tokenized/tokenizer')

    print('Loading training config...')
    run_name='tlm-{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    training_args = TrainingArguments(
        output_dir='./{}'.format(run_name),
        num_train_epochs=2,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=64,
        warmup_steps=10000,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        bf16=True,
        report_to='wandb',
        run_name=run_name,
    )

    '''
    #100M
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )

    #300M
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=512,
    )

    #1B
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=1280,
        num_hidden_layers=36,
        num_attention_heads=20,
        intermediate_size=5120,
        max_position_embeddings=512,
    )

    model = BertForMaskedLM(config)
    '''

    print('Loading model...')
    model = BertForMaskedLM.from_pretrained('bert-large-uncased')
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForTemporalSpanMasking(tokenizer, num_spans=55)

    print('Loading trainer...')
    trainer = TLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print('Training!')
    trainer.train()

    print('Saving model...')
    trainer.save_model('./model')