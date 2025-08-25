import os
import json
import glob
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer


# general file preproc
def preproc_general(fpath):

    with open(fpath, 'r', encoding='utf-8') as file:
        text = file.read()

    num_texts = 0
    year = int(fpath.split('/')[-1].split('-')[0])
    month = int(fpath.split('/')[-1].split('-')[1])
    print('parsed year and month', year, month, 'from', fpath)
    text = text.replace('<p>', ' ')
    text = text.replace('<h>', ' ')
    text = text.replace('@!', ' ')
    text = text.replace('@ @ @ @ @ @ @ @ @ @', '[MASK_NOLOSS]')
    text = text.split('@@')
    for e in text:
        if e.strip():
            # Find the first whitespace and split into 2 parts
            first_space_index = e.find(' ')
            if first_space_index != -1:
                docid = e[:first_space_index]
                text = e[first_space_index + 1:]
                num_texts += 1
                yield text, year, month
    print(f"Extracted {num_texts} texts from {fpath}")

# preproc orchestration
def preproc_file(fpath):
    return preproc_general(fpath)


def tokenize_source(source, tokenizer):
    sequence_length = 512 - 1 # -1 to make room for the year token
    overlap = 32
    step_size = sequence_length - overlap
    all_sequences = []
    for text, year, month in preproc_file(source):
        timestr = '[TIME:{i}-{j}]'.format(i=year, j=month)
        time_token = tokenizer.encode(timestr, add_special_tokens=False)[0]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        cur_sequences = [[time_token] + tokens[i:i + sequence_length] for i in range(0, len(tokens), step_size)]

        all_sequences.extend(cur_sequences)
    return all_sequences


if __name__ == "__main__":
    print('Creating tokenizer...')
    #create the tokenizer
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
    tokenizer.save_pretrained('./now_tokenized_ettin_large/tokenizer/')

    # get all the english sources
    raw_sources = glob.glob('./now/text/*us*.txt')

    # tokenize the sources
    print('Tokenizing sources...')
    for source in tqdm(raw_sources):
        sequences = tokenize_source(source, tokenizer)

        # write sequences to JSONL file (one sequence per line)
        output_file = f'./now_tokenized_ettin_large/{os.path.basename(source).split(".")[0]}.jsonl'
        with open(output_file, 'w') as f:
            for sequence in sequences:
                json.dump(sequence, f)
                f.write('\n')

