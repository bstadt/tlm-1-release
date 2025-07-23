from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
from loader import COCATokenizedDataset, DataCollatorForTemporalSpanMasking

model_name = 'jhu-clsp/ettin-encoder-1b'
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

additional_special_tokens = ['[MASK_NOLOSS]'] + ['[YEAR:{i}]'.format(i=i) for i in range(1900, 2025)]
special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Set new special token embeddings to the embedding of the space character
space_token_id = tokenizer.convert_tokens_to_ids(' ')
if space_token_id == tokenizer.unk_token_id:
    # If the space is not a single token, try using the most common whitespace token
    space_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

embedding = model.get_input_embeddings()
new_token_ids = tokenizer.convert_tokens_to_ids(additional_special_tokens)
space_embedding = embedding.weight.data[space_token_id].clone()
for token_id in new_token_ids:
    embedding.weight.data[token_id] = space_embedding

dataset = COCATokenizedDataset(root_path='./coca_tokenized_ettin_large', debug=False)
data_collator = DataCollatorForTemporalSpanMasking(tokenizer, num_spans=55)

dataloader = DataLoader(dataset, batch_size=64, collate_fn=data_collator)

for batch in dataloader:

    # Replace input_ids and labels with unk_token_id if not in tokenizer
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    vocab_size = len(tokenizer)
    unk_token_id = tokenizer.unk_token_id

    # Create a mask for tokens not in the vocab
    invalid_input_mask = input_ids >= vocab_size
    invalid_label_mask = (labels >= vocab_size) & (labels != -100)  # don't touch ignore index

    if unk_token_id is not None:
        input_ids[invalid_input_mask] = unk_token_id
        labels[invalid_label_mask] = unk_token_id

    batch["input_ids"] = input_ids
    batch["labels"] = labels

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch["labels"]

        # Only consider positions where label != -100 (i.e., masked positions)
        mask = labels != -100
        if mask.sum() == 0:
            print("No masked positions in this batch.")
        else:
            preds = logits.argmax(dim=-1)
            correct = (preds == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            print(f"Infill accuracy: {accuracy:.4f} ({correct.sum().item()}/{mask.sum().item()})")
    break