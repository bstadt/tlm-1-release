import pdb
from torch.utils.data import DataLoader, Dataset
from loader import DataCollatorForSpanMasking, COCATokenizedDataset
from transformers import BertTokenizerFast, TrainingArguments, Trainer, BertConfig, BertForMaskedLM

if __name__ == "__main__":
    
    print('Loading dataset...')
    dataset = COCATokenizedDataset(root_path='./coca_tokenized')

    print('Loading tokenizer...')
    tokenizer = BertTokenizerFast.from_pretrained('./coca_tokenized/tokenizer')

    print('Loading training config...')
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=64,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )

    print('Loading model...')
    model = BertForMaskedLM(config)

    print('Loading trainer...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSpanMasking(tokenizer),
    )

    print('Training!')
    trainer.train()

    print('Saving model...')
    trainer.save_model('./model')