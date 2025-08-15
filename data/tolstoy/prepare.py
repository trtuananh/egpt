import os
import requests
from tqdm import tqdm
import numpy as np
import tiktoken
import re

# Function to download a book from Project Gutenberg
def download_book(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Downloaded {filename}")

# Function to clean Gutenberg header, footer, and table of contents
def clean_gutenberg_text(text, book_title):
    # Remove header and footer
    start_marker = f"*** START OF THE PROJECT GUTENBERG EBOOK {book_title.upper()} ***"
    end_marker = f"*** END OF THE PROJECT GUTENBERG EBOOK {book_title.upper()} ***"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx]
    
    # Clean extra newlines and whitespace
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text

if __name__ == '__main__':
    # Step 1: Download the books
    books = {
        'war_and_peace.txt': {
            'url': 'https://www.gutenberg.org/files/2600/2600-0.txt',
            'title': 'WAR AND PEACE',
            'id': 'WAR AND PEACE',
            'split': 'train'
        },
        'anna_karenina.txt': {
            'url': 'https://www.gutenberg.org/files/1399/1399-0.txt',
            'title': 'ANNA KARENINA',
            'id': '1399',
            'split': 'train'
        },
        'moby_dick.txt': {
            'url': 'https://www.gutenberg.org/files/2701/2701-0.txt',
            'title': 'MOBY DICK; OR THE WHALE',
            'id': '2701',
            'split': 'val'
        }
    }

    for filename, info in books.items():
        download_book(info['url'], filename)

    # Step 2: Load and clean the texts
    texts = {}
    for filename, info in books.items():
        with open(filename, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        texts[filename] = clean_gutenberg_text(raw_text, info['id'])
        print(f"Cleaned {filename} length: {len(texts[filename])} characters")
        print(texts[filename][:100])

    # Step 3: Tokenize using GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    train_ids = []
    val_ids = []

    # Process train books (War and Peace + Anna Karenina, joined by EOT)
    for filename in ['war_and_peace.txt', 'anna_karenina.txt']:
        ids = enc.encode_ordinary(texts[filename])
        train_ids.extend(ids)
        train_ids.append(enc.eot_token)  # Append EOT after each book
    print(f"Train tokens: {len(train_ids)}")

    # Process val book (Moby Dick)
    val_ids = enc.encode_ordinary(texts['moby_dick.txt'])
    val_ids.append(enc.eot_token)  # Append EOT at the end
    print(f"Val tokens: {len(val_ids)}")

    # Step 4: Save to .bin files using memmap
    for split, split_ids in [('train', train_ids), ('val', val_ids)]:
        arr_len = len(split_ids)
        bin_filename = f"{split}.bin"
        dtype = np.uint16  # Since GPT-2 max_token_value < 2**16
        arr = np.memmap(bin_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        arr[:] = split_ids  # Write the ids directly
        arr.flush()
        print(f"Saved {bin_filename} with {arr_len} tokens")

    # Optional: Basic statistics
    print(f"Train average tokens per 'document': {len(train_ids)} (two books)")
    print(f"Val average tokens per 'document': {len(val_ids)} (single book)")
    