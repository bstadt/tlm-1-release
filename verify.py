import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from loader import DataCollatorForSpanMasking, COCATokenizedDataset
from torch.utils.data import DataLoader

# Load the saved model and tokenizer
model = BertForMaskedLM.from_pretrained('./model')
tokenizer = BertTokenizerFast.from_pretrained('./model')

dataset = COCATokenizedDataset(root_path='./coca_tokenized', debug=True)

# Create dataloader with custom collator
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=DataCollatorForSpanMasking(tokenizer)
)

# Get the next batch
batch = next(iter(dataloader))

# Move batch to device (assuming CPU for now)
input_ids = batch['input_ids']
labels = batch['labels']

# Get model predictions
with torch.no_grad():
    outputs = model(input_ids=input_ids)
    predictions = outputs.logits

# Get predicted tokens (argmax)
predicted_tokens = torch.argmax(predictions, dim=-1)

# Process multiple examples from batch
num_examples = min(5, len(input_ids))  # Show up to 10 examples

for example_idx in range(num_examples):
    example_input = input_ids[example_idx]
    example_labels = labels[example_idx]
    example_predicted = predicted_tokens[example_idx]

    
    # Convert to tokens
    original_tokens = tokenizer.decode(example_input)
    ground_truth_tokens = tokenizer.decode(example_labels)
    predicted_tokens_list = tokenizer.decode(example_predicted)
    
    print(original_tokens)
    print()
    print(ground_truth_tokens)
    print()
    print(predicted_tokens_list)
    print('-' * 80)