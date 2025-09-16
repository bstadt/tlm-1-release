import sys
import click
sys.path.append('../')
from loader import JSONLTokenizedDataset
dataset = JSONLTokenizedDataset(root_path='../coca_tokenized_ettin_large', debug=False)

from transformers import AutoModelForMaskedLM, AutoTokenizer
# Load the saved model and tokenizer
#loadstr = '/home/ubuntu/bstadt-tlm/tlm/now-tlm-2025-08-15_15-08-29/checkpoint-112712/'
loadstr = '/home/bstadt/root/tlm/models/tlm-2025-08-05_16-42-11/checkpoint-10500/'
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


def main(phrase):
    """Mine uses of a target phrase from the dataset."""
    print(f"Mining uses of phrase: '{phrase}'")
    
    uses = mine_uses(phrase, tokenizer)
    
    # Create output filename based on the target phrase
    safe_phrase = phrase
    output_filename = f'{safe_phrase}_uses.txt'
    
    # Write the uses to a file
    print(f"Found {len(uses)} uses. Writing to {output_filename}")
    with open(output_filename, 'w') as f:
        for use in uses:
            f.write(use + '\n')
    
    print(f"Complete! Results saved to {output_filename}")

if __name__ == '__main__':
    main('good')
