import sys
sys.path.append('../')
from loader import JSONLTokenizedDataset
dataset = JSONLTokenizedDataset(root_path='../now_tokenized_ettin_large', debug=False)

from transformers import AutoModelForMaskedLM, AutoTokenizer
# Load the saved model and tokenizer
#loadstr = '/home/ubuntu/bstadt-tlm/tlm/now-tlm-2025-08-15_15-08-29/checkpoint-112712/'
loadstr = '/home/ubuntu/bstadt-tlm/tlm/now-tlm-2025-08-15_15-08-29/checkpoint-60000/'
model = AutoModelForMaskedLM.from_pretrained(loadstr)
#tokenizer = BertTokenizerFast.from_pretrained('../coca_tokenized/tokenizer/')
tokenizer = AutoTokenizer.from_pretrained(loadstr)

from tqdm import tqdm
def mine_uses(phrase, tokenizer):
    uses = []
    for elem in tqdm(dataset):
        decoded = tokenizer.decode(elem['input_ids'])
        if phrase in decoded:
            uses.append(decoded)
    return uses


uses = mine_uses('vibe', tokenizer)

# Write the uses to a file
with open('vibe_uses.txt', 'w') as f:
    for use in uses:
        f.write(use + '\n')
