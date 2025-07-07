import pdb
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizerFast
from loader import DataCollatorForTemporalSpanMasking, COCATokenizedDataset
from torch.utils.data import DataLoader

def compute_pppl(text, model, tokenizer):
    model.eval()

    encoded = tokenizer(text, return_tensors='pt')
    input_ids = encoded['input_ids'][0]
    seq_len = input_ids.size(0)

    positions = list(range(0, seq_len))  # ignore [CLS] and [SEP]
    n_positions = len(positions)

    input_ids_expanded = input_ids.repeat(n_positions, 1)
    for i, pos in enumerate(positions):
        input_ids_expanded[i, pos] = tokenizer.mask_token_id


    with torch.no_grad():
        outputs = model(input_ids=input_ids_expanded)
        logits = outputs.logits

    target_tokens = input_ids[positions]  # true tokens
    masked_logits = logits[range(n_positions), positions]
    log_probs = F.log_softmax(masked_logits, dim=-1)
    token_log_probs = log_probs[range(n_positions), target_tokens]

    avg_log_likelihood = token_log_probs.mean()
    return torch.exp(-avg_log_likelihood).item()  # Pseudo-perplexity


def predict_masked_tokens(text, tokenizer, model, top_k=5):
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits
    mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    results = []
    for idx in mask_token_index:
        logits = predictions[0, idx]
        top_tokens = torch.topk(logits, top_k).indices
        tokens = [tokenizer.decode([token]) for token in top_tokens]
        results.append(tokens)

    return results

# Load the saved model and tokenizer
model = BertForMaskedLM.from_pretrained('./100m/lr-4e4_4-epoch_loss-3.1_model/')
tokenizer = BertTokenizerFast.from_pretrained('./coca_tokenized/tokenizer/')

dataset = COCATokenizedDataset(root_path='./coca_tokenized', debug=True)

# Create dataloader with custom collator
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=DataCollatorForTemporalSpanMasking(tokenizer)
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
    pdb.set_trace()
    original_tokens = tokenizer.decode(example_input)
    ground_truth_tokens = tokenizer.decode(example_labels)
    predicted_tokens_list = tokenizer.decode(example_predicted)
    
    print(original_tokens)
    print()
    print(ground_truth_tokens)
    print()
    print(predicted_tokens_list)
    print('-' * 80)