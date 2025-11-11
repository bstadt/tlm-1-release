swadesh_words = [
    "I",
    "you",
    "we",
    "this",
    "that",
    "who",
    "what",
    "not",
    "all",
    "many",
    "one",
    "two",
    "big",
    "long",
    "small",
    "woman",
    "man",
    "person",
    "fish",
    "bird",
    "dog",
    "louse",
    "tree",
    "seed",
    "leaf",
    "root",
    "bark",
    "skin",
    "flesh",
    "blood",
    "bone",
    "grease",
    "egg",
    "horn",
    "tail",
    "feather",
    "hair",
    "head",
    "ear",
    "eye",
    "nose",
    "mouth",
    "tooth",
    "tongue",
    "claw",
    "foot",
    "knee",
    "hand",
    "belly",
    "neck",
    "breast",
    "heart",
    "liver",
    "drink",
    "eat",
    "bite",
    "see",
    "hear",
    "know",
    "sleep",
    "die",
    "kill",
    "swim",
    "fly",
    "walk",
    "come",
    "lie",
    "sit",
    "stand",
    "give",
    "say",
    "sun",
    "moon",
    "star",
    "water",
    "rain",
    "stone",
    "sand",
    "earth",
    "cloud",
    "smoke",
    "fire",
    "ash",
    "burn",
    "path",
    "mountain",
    "red",
    "green",
    "yellow",
    "white",
    "black",
    "night",
    "hot",
    "cold",
    "full",
    "new",
    "good",
    "round",
    "dry",
    "name",
]

# Custom words list
custom_words = [
    "cell",
    "vibe", 
    "seven",
]

number_words = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]

value_words = [
    "true",
    "false",
    "right",
    "wrong"
    "moral",
    "immoral",
    "justified",
    "unjustified",
    "good",
    "bad",
    "ethical",
    "unethical",
    "crime",
    "sin"
]


import sys
import csv
import re
sys.path.append('../')
from loader import JSONLTokenizedDataset
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

def load_semantic_change_words():
    """Load words from the semantic change probe list CSV."""
    try:
        semantic_words = []
        with open('semantic_change_probe_list.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                word = row['Word'].strip()
                if word and word not in semantic_words:  # Avoid duplicates
                    semantic_words.append(word)
        
        print(f"Loaded {len(semantic_words)} words from semantic_change_probe_list.csv")
        return semantic_words
    except FileNotFoundError:
        print("Warning: semantic_change_probe_list.csv not found. Skipping semantic change words.")
        return []
    except Exception as e:
        print(f"Error loading semantic change words: {e}")
        return []

def is_word_match(word, text):
    """
    Check if a word appears as a complete word (not substring) in text.
    Uses word boundaries - checks for punctuation or space after the word.
    """
    # Create regex pattern for word boundaries
    # \b matches word boundaries, but we also want to be more explicit about what follows
    pattern = r'\b' + re.escape(word) + r'(?=\s|[^\w]|$)'
    return bool(re.search(pattern, text, re.IGNORECASE))

def mine_all_words():
    """Mine uses of all word lists in a single pass through the dataset."""
    
    # Load semantic change words from CSV
    semantic_words = load_semantic_change_words()
    
    # Combine all word lists
    #all_words = swadesh_words + custom_words + semantic_words + value_words + number_words
    all_words = ['crunchy']
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    print(f"Total unique words to mine: {len(unique_words)}")
    print(f"  - Swadesh words: {len(swadesh_words)}")
    print(f"  - Custom words: {len(custom_words)}")
    print(f"  - Semantic change words: {len(semantic_words)}")
    print(f"  - Value words: {len(value_words)}")
    print(f"  - Number words: {len(number_words)}")
    
    # Initialize dataset and tokenizer
    print("Loading dataset and tokenizer...")
    dataset = JSONLTokenizedDataset(root_path='../coca_tokenized_ettin_large', debug=False)
    loadstr = '/home/bstadt/root/tlm/models/tlm-2025-08-05_16-42-11/checkpoint-10500/'
    tokenizer = AutoTokenizer.from_pretrained(loadstr)
    
    # Initialize results dictionary
    word_uses = defaultdict(list)
    
    print(f"Starting single-pass mining for {len(unique_words)} words...")
    
    # Single pass through the dataset
    for elem in tqdm(dataset, desc="Processing dataset"):
        decoded = tokenizer.decode(elem['input_ids'])
        
        # Check for all words in this decoded text using proper word boundary matching
        for word in unique_words:
            if is_word_match(word, decoded):
                word_uses[word].append(decoded)
    
    return word_uses

def write_all_results(word_uses):
    """Write results for all words to their respective files."""
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "mined_usages"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nWriting results to files in {output_dir}/ directory...")
    
    for word, uses in word_uses.items():
        # Create safe filename
        safe_word = word.replace('/', '_').replace(' ', '_').strip('_')
        output_filename = os.path.join(output_dir, f'{safe_word}_uses.txt')
        
        print(f"Found {len(uses)} uses for '{word}'. Writing to {output_filename}")
        
        with open(output_filename, 'w') as f:
            for use in uses:
                f.write(use + '\n')
    
    print(f"\nComplete! Results saved for {len(word_uses)} words in {output_dir}/ directory.")

def print_summary(word_uses):
    """Print a summary of findings."""
    print(f"\n{'='*50}")
    print("MINING SUMMARY")
    print(f"{'='*50}")
    
    # Load semantic words to get total count
    semantic_words = load_semantic_change_words()
    all_words = swadesh_words + custom_words + semantic_words
    unique_all_words = list(dict.fromkeys(all_words))  # Remove duplicates preserving order
    
    total_uses = sum(len(uses) for uses in word_uses.values())
    print(f"Total uses found: {total_uses}")
    print(f"Words with uses: {len(word_uses)}")
    print(f"Total words searched: {len(unique_all_words)}")
    print(f"  - Swadesh words: {len(swadesh_words)}")
    print(f"  - Custom words: {len(custom_words)}")
    print(f"  - Semantic change words: {len(semantic_words)}")
    
    # Show top 10 most frequent words
    sorted_words = sorted(word_uses.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\nTop 10 most frequent words:")
    for i, (word, uses) in enumerate(sorted_words[:10], 1):
        print(f"{i:2d}. '{word}': {len(uses)} uses")
    
    # Show words with no uses by category
    swadesh_no_uses = [word for word in swadesh_words if word not in word_uses]
    custom_no_uses = [word for word in custom_words if word not in word_uses]
    semantic_no_uses = [word for word in semantic_words if word not in word_uses]
    
    if swadesh_no_uses:
        print(f"\nSwadesh words with no uses found ({len(swadesh_no_uses)}):")
        for word in swadesh_no_uses:
            print(f"  - '{word}'")
    
    if custom_no_uses:
        print(f"\nCustom words with no uses found ({len(custom_no_uses)}):")
        for word in custom_no_uses:
            print(f"  - '{word}'")
    
    if semantic_no_uses:
        print(f"\nSemantic change words with no uses found ({len(semantic_no_uses)}):")
        for word in semantic_no_uses[:10]:  # Show only first 10 to avoid clutter
            print(f"  - '{word}'")
        if len(semantic_no_uses) > 10:
            print(f"  ... and {len(semantic_no_uses) - 10} more")

if __name__ == '__main__':
    try:
        # Mine all words in a single pass
        word_uses = mine_all_words()
        
        # Write results to files
        write_all_results(word_uses)
        
        # Print summary
        print_summary(word_uses)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise