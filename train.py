from torch.utils.data import DataLoader, Dataset
from loader import DataCollatorForTemporalSpanMasking, COCATokenizedDataset
from transformers import BertTokenizerFast, TrainingArguments, Trainer, BertConfig, BertForMaskedLM

if __name__ == "__main__":
    
    print('Loading dataset...')
    dataset = COCATokenizedDataset(root_path='./coca_tokenized', debug=False)

    print('Loading tokenizer...')
    tokenizer = BertTokenizerFast.from_pretrained('./coca_tokenized/tokenizer')

    print('Loading training config...')
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=128,
        warmup_steps=1000,
        learning_rate=4e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    #100M
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )

    '''

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
    '''

    print('Loading model...')
    model = BertForMaskedLM(config)

    data_collator = DataCollatorForTemporalSpanMasking(tokenizer)

    print('Loading trainer...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print('Training!')
    trainer.train()

    print('Saving model...')
    trainer.save_model('./model')