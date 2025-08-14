# Script to download, process, tokenize, and save "War and Peace" as train.bin and val.bin
# Similar to the OpenWebText processing script

import os
import requests
from tqdm import tqdm
import numpy as np
import tiktoken
import re

# Function to download the book from Project Gutenberg
def download_book(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Downloaded {filename}")

# Function to clean Gutenberg header and footer
def clean_gutenberg_text(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx]
    return text.strip()

if __name__ == '__main__':
    # Step 1: Download the book
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"
    filename = "war_and_peace.txt"
    download_book(url, filename)

    # Step 2: Load and clean the text
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    cleaned_text = clean_gutenberg_text(raw_text)
    print(f"Cleaned text length: {len(cleaned_text)} characters")

    # Step 3: Tokenize using GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(cleaned_text)  # Encode the entire book
    ids.append(enc.eot_token)  # Append EOT at the end
    total_tokens = len(ids)
    print(f"Total tokens: {total_tokens}")

    # Step 4: Split into train (80% first) and val (20% last)
    split_idx = int(0.8 * total_tokens)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]
    print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

    # Step 5: Save to .bin files using memmap
    for split, split_ids in [('train', train_ids), ('val', val_ids)]:
        arr_len = len(split_ids)
        bin_filename = f"{split}.bin"
        dtype = np.uint16  # Since GPT-2 max_token_value < 2**16
        arr = np.memmap(bin_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        arr[:] = split_ids  # Write the ids directly
        arr.flush()
        print(f"Saved {bin_filename} with {arr_len} tokens")

    # Optional: Basic statistics
    print(f"Train average tokens per 'document': {len(train_ids)} (single book)")
    print(f"Val average tokens per 'document': {len(val_ids)} (single book)")

    # To read later: m = np.memmap('train.bin', dtype=np.uint16, mode='r')
