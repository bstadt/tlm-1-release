from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from preproc import preproc_general

model_name = 'jhu-clsp/ettin-encoder-400m'
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


for text, year in preproc_general('coca/text/text_acad_1996.txt'):
    encoding = tokenizer(text, return_tensors='pt')
    input_ids = encoding['input_ids']

    # Manually mask 15% of the input ids
    masked_input_ids = input_ids.clone()
    probability_matrix = torch.full(masked_input_ids.shape, 0.15)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_input_ids[masked_indices] = tokenizer.mask_token_id
    print('masked_input_ids:', masked_input_ids[0, :20])

    labels = input_ids.clone()
    outputs = model(masked_input_ids)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    print('predictions:', predictions[0, :20])
    print('labels:', labels[0, :20])
    #mask = labels != -100
    mask = masked_indices
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total}")
    exit()

