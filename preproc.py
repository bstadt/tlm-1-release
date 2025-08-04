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
    year = fpath.split('/')[-1].split('_')[-1].split('.')[0]
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
                yield text, year
    print(f"Extracted {num_texts} texts from {fpath}")

# blog and web preproc
print('Loading filenum_to_year...')
filenum_to_year = {}
with open('./coca/sources.txt', 'r', encoding='utf-8', errors='ignore') as file:
    for line in file:
        parts = line.split('\t')
        try:
            filenum_to_year[parts[0]] = int(parts[1])
        except Exception as e:
            print(parts)

def preproc_blog_web(fpath):

    with open(fpath, 'r', encoding='utf-8') as file:
        text = file.read()

    num_texts = 0

    text = text.replace('<p>', ' ')
    text = text.replace('<h>', ' ')
    text = text.replace('&', ' ')
    text = text.replace('@ @ @ @ @ @ @ @ @ @', '[MASK_NOLOSS]')
    text = text.split('@@')
    for e in text:
        if e.strip():
            # Find the first whitespace and split into 2 parts
            first_space_index = e.find(' ')
            if first_space_index != -1:
                docid = e[:first_space_index]
                text = e[first_space_index + 1:]
                if docid in filenum_to_year:
                    year = filenum_to_year[docid]
                    num_texts += 1
                    yield text, year
                else:
                    print(f"docid {docid} not found in filenum_to_year when processing file {fpath} (num_texts: {num_texts})")
    print(f"Extracted {num_texts} texts from {fpath}")

# preproc orchestration
def preproc_file(fpath):

    if 'acad' in fpath:
        texts = preproc_general(fpath)
    elif 'blog' in fpath:
        texts = preproc_blog_web(fpath)
    elif 'fic' in fpath:
        texts = preproc_general(fpath)
    elif 'mag' in fpath:
        texts = preproc_general(fpath)
    elif 'news' in fpath:
        texts = preproc_general(fpath)
    elif 'spok' in fpath:
        texts = preproc_general(fpath)
    elif 'tvm' in fpath:
        texts = preproc_general(fpath)
    elif 'web' in fpath:
        texts = preproc_blog_web(fpath)


    else:                
        raise NotImplementedError(f"Genre type not supported: {fpath}")

    return texts

def tokenize_source(source, tokenizer):
    sequence_length = 512 - 1 # -1 to make room for the year token
    overlap = 128
    step_size = sequence_length - overlap
    all_sequences = []
    for text, year in preproc_file(source):
        year_token = tokenizer.encode('[YEAR:{i}]'.format(i=year), add_special_tokens=False)[0]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        cur_sequences = [[year_token] + tokens[i:i + sequence_length] for i in range(0, len(tokens), step_size)]

        all_sequences.extend(cur_sequences)
    return all_sequences


if __name__ == "__main__":
    print('Creating tokenizer...')
    #create the tokenizer
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-400m")

    # Add new special tokens
    additional_special_tokens = ['[MASK_NOLOSS]'] + ['[YEAR:{i}]'.format(i=i) for i in range(1990, 2025)]
    special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # save the tokenizer
    os.makedirs('./coca_tokenized_ettin_large/tokenizer/', exist_ok=True)
    tokenizer.save_pretrained('./coca_tokenized_ettin_large/tokenizer/')

    # get all the sources
    raw_sources = glob.glob('./coca/text/text*.txt')

    # tokenize the sources
    print('Tokenizing sources...')
    for source in tqdm(raw_sources):
        sequences = tokenize_source(source, tokenizer)

        # write sequences to JSONL file (one sequence per line)
        output_file = f'./coca_tokenized_ettin_large/{os.path.basename(source).split(".")[0]}.jsonl'
        with open(output_file, 'w') as f:
            for sequence in sequences:
                json.dump(sequence, f)
                f.write('\n')

