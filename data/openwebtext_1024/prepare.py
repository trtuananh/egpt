import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import torch

MINIMUM_LENGTH = 1024  # Minimum length of sequences to keep

# number of workers in .map() call
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

# Directory for data storage
data_dir = os.path.dirname(__file__)

if __name__ == '__main__':
    # Load dataset
    dataset = load_dataset(
        "Skylion007/openwebtext",
        trust_remote_code=True,
        revision="refs/convert/parquet",
        num_proc=num_proc_load_dataset
    )

    # Tokenization function
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize dataset
    tokenized = dataset["train"].map(
        process,
        remove_columns=['text'],
        desc="tokenizing the dataset",
        num_proc=num_proc,
    )

    # Filter rows with length > 16384 tokens
    filtered_dataset = tokenized.filter(
        lambda example: example['len'] > MINIMUM_LENGTH,
        num_proc=num_proc,
        desc="filtering long sequences"
    )

    # Create train/val split after filtering
    split_dataset = filtered_dataset.train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    import matplotlib.pyplot as plt

    # Plot histogram of lengths for train split
    lengths = split_dataset['train']['len']

    # In ra một vài thống kê cơ bản
    import numpy as np
    print(f"Tổng số sample: train {len(lengths)}, val {len(split_dataset['val']['len'])}")
    print(f"Tổng số token: train {np.sum(lengths)}, val {np.sum(split_dataset['val']['len'])}")
    print(f"Độ dài trung bình: {np.mean(lengths):.2f}")
    print(f"Độ dài trung vị: {np.percentile(lengths, 25)}, {np.median(lengths)}, {np.percentile(lengths, 75)}")
    print(f"Độ dài tối thiểu: {np.min(lengths)}")
    print(f"Độ dài tối đa: {np.max(lengths)}")

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=100, range=(MINIMUM_LENGTH, max(lengths)), log=True)
    plt.title('Phân bố độ dài Token trong OpenWebText (Train Split, >16384 tokens)')
    plt.xlabel('Độ dài Token')
    plt.ylabel('Số lượng (log scale)')
    plt.show()

    # Print statistics
    print(f"Tổng số sample (train): {len(lengths)}")
    print(f"Độ dài trung bình (train): {np.mean(lengths):.2f}")
    print(f"Độ dài trung vị (train): {np.median(lengths)}")
    print(f"Độ dài tối đa (train): {np.max(lengths)}")
    print(f"Tổng số sample (val): {len(split_dataset['val']['len'])}")

    # Save row metadata (start positions and lengths) for random access
    for split, dset in split_dataset.items():
        filename = os.path.join(data_dir, f'{split}.bin')
        meta_filename = os.path.join(data_dir, f'{split}_meta.npy')
        
        # Calculate total length and create memmap
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Store start positions and lengths for each row
        start_positions = []
        lengths = []
        idx = 0
        
        # Write tokens and collect metadata
        num_samples = len(dset)
        total_batches = min(1024, num_samples)
        
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            for ids in batch['ids']:
                arr[idx:idx + len(ids)] = ids
                start_positions.append(idx)
                lengths.append(len(ids))
                idx += len(ids)
        arr.flush()
        
        # Save metadata
        np.save(meta_filename, {'start_positions': start_positions, 'lengths': lengths})
