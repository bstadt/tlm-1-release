import pdb
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, Dataset
from loader import DataCollatorForSpanMasking, COCATokenizedDataset

if __name__ == "__main__":
    dataset = COCATokenizedDataset(root_path='./coca_tokenized')
    tokenizer = BertTokenizerFast.from_pretrained('./coca_tokenized/tokenizer')
    loader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorForSpanMasking(tokenizer))