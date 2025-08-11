import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torch
import linecache
import json
from typing import Any, Dict, List

class JSONLTokenizedDataset(Dataset):
    def __init__(self, root_path, debug=False):
        self.root_path = root_path
        if debug:
            self.source_files = glob.glob(os.path.join(root_path, '*acad*.jsonl'))
        else:
            self.source_files = glob.glob(os.path.join(root_path, '*.jsonl'))
            self.source_files = [e for e in self.source_files if 'web' not in e]
            self.source_files = [e for e in self.source_files if 'blog' not in e]
        
        # Compute total length and file lengths
        self.file_lengths = []
        for file_path in self.source_files:
            with open(file_path, 'r') as f:
                file_length = sum(1 for _ in f)
                print('Loaded file', file_path, 'with length', file_length)
                self.file_lengths.append(file_length)
        
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.total_length = self.cumulative_lengths[-1]
        
    def __len__(self):
        return self.total_length

    def get_source_file_from_item_idx(self, idx):
        linenum = idx + 1 #since linecache is 1-indexed for some reason
        file_idx = np.searchsorted(self.cumulative_lengths, linenum)
        return self.source_files[file_idx]

    def __getitem__(self, idx):

        linenum = idx + 1 #since linecache is 1-indexed for some reason
        file_idx = np.searchsorted(self.cumulative_lengths, linenum)
        if file_idx == 0:
            line_idx = linenum
        else:
            line_idx = linenum - self.cumulative_lengths[file_idx-1]

        line = linecache.getline(self.source_files[file_idx], line_idx)
        if not line:
            raise IndexError(f"Index {linenum} out of bounds for file {self.source_files[file_idx]} with length {self.file_lengths[file_idx]}")
            
        data = json.loads(line.strip())
        return {'input_ids': torch.tensor(data)}

class DataCollatorForTemporalSpanMasking:
    def __init__(self, tokenizer, num_spans=30, p_time_mask=0.9):
        self.tokenizer = tokenizer
        self.p_time_mask = p_time_mask
        self.num_spans = num_spans
        self.geom_p = 0.2
        self.lower = 1
        self.upper = 4
        self.lens = list(range(self.lower, self.upper + 1))
        self.len_distrib = [self.geom_p * (1-self.geom_p)**(i - self.lower) for i in self.lens] if self.geom_p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        self.num_spans = num_spans

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        # Apply masking
        masked_inputs = []
        labels = []

        for example in batch:

            #NOTE this is kinda a hack but it affects very few sequences at the end of the documents
            #It should really be dealt with in the preproc step
            #It will result in some batches having less than batch_size examples
            if example['input_ids'].shape[0] != 512:
                continue

            input_ids = example['input_ids']
            cur_labels = input_ids.clone()

            num_spans = self.num_spans

            # Time masking
            if np.random.random() < self.p_time_mask:
                input_ids[0] = self.tokenizer.mask_token_id

            # Span masking
            span_lens = np.random.choice(self.lens, num_spans, p=self.len_distrib)
            for span_len in span_lens:
                start_idx = np.random.randint(0, input_ids.shape[0] - span_len)
                end_idx = start_idx + span_len
                input_ids[start_idx:end_idx] = self.tokenizer.mask_token_id
            masked_inputs.append(input_ids)
            labels.append(cur_labels)
        input_ids = torch.stack(masked_inputs)
        labels = torch.stack(labels)

        # Set non mask tokens to -100 so they are ignored by the loss function
        labels[input_ids != self.tokenizer.mask_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
