# EGPT: Encoder-Enhanced GPT for Extended Context Length

## Overview

EGPT (Encoder-Enhanced GPT) is a novel architecture that extends the context length of traditional transformer models by leveraging a pre-trained encoder (such as GPT-2) to extract features from input sequences. This approach enables exponential scaling of context window size while maintaining computational efficiency.

## Architecture

### Core Components

The EGPT model consists of several key components:

1. **Frozen Pre-trained Encoder**: A GPT-based encoder (e.g., GPT-2) that extracts contextual features from input tokens
2. **Feature Embeddings**: 
   - Position embeddings for token positions within chunks
   - Length embeddings for chunk sequence positions
3. **Transformer Decoder Layers**: Standard transformer blocks that process the encoded features
4. **Language Model Head**: Final layer for token prediction

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           EGPT Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [tok1, tok2, ..., tokN] (N >> 1024)                     │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Chunk Division                             │    │
│  │  [chunk1] [chunk2] ... [chunkK]                         │    │
│  │  (1024)   (1024)       (≤1024)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Frozen GPT-2 Encoder                          │    │
│  │  chunk1 → features1 (B, T, C)                           │    │
│  │  chunk2 → features2 (B, T, C)                           │    │
│  │  ...                                                    │    │
│  │  chunkK → featuresK (B, T, C)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Feature Combination                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ Previous    │  │ Position    │  │ Length      │      │    │
│  │  │ Features    │  │ Embedding   │  │ Embedding   │      │    │
│  │  │ (cached)    │  │             │  │             │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │         │               │               │               │    │
│  │         └───────────────┼───────────────┘               │    │
│  │                         ▼                               │    │
│  │              Combined Features                          │    │
│  │              (B, T, N, C)                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EGPT Decoder Layers                        │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Multi-Head Self-Attention                       │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Feed Forward Network                            │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │  │ (Repeated for n_layer times)                    │    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Language Model Head                        │    │
│  │              Output Logits                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### Context Extension Mechanism

1. **Input Chunking**: Input tokens are divided into chunks of size equal to the encoder's context window (e.g., 1024 tokens for GPT-2)

2. **Feature Extraction**: Each chunk is processed by the frozen pre-trained encoder to extract high-level contextual features

3. **Hierarchical Processing**: The EGPT decoder processes relationships between chunk features, enabling understanding of long-range dependencies

4. **Exponential Scaling**: The effective context length becomes `encoder_block_size × egpt_block_size` (e.g., 1024 × 16 = 16,384 tokens)

## Data Preparation

### Dataset

The model is evaluated on long-context datasets such as the complete text of "War and Peace", which provides an ideal testbed for long-range dependency modeling.

### Data Split

- **Training**: 80% of the dataset
- **Validation**: 20% of the dataset

### Setup

Run the data preparation script:

```bash
python prepare.py
```

This script will:
- Download the War and Peace dataset
- Process and tokenize the text
- Split into training and validation sets
- Save processed data for training

## Model Variants

### EGPT (Encoder GPT)
- **Context Length**: 1024 × 16 = 16,384 tokens
- **Architecture**: Hierarchical processing with chunk-based encoding
- **Encoder**: Frozen GPT-2
- **Decoder Layers**: 3 transformer blocks

### DeepGPT (Baseline)
- **Context Length**: 1024 tokens (standard)
- **Architecture**: Traditional transformer without context extension
- **Encoder**: Frozen GPT-2
- **Decoder Layers**: 3 transformer blocks

## Results

### Performance Comparison

| Model | Train Loss | Validation Loss | Context Length |
|-------|------------|-----------------|----------------|
| **EGPT** | **2.1431** | **3.3893** | 16,384 tokens |
| DeepGPT | 3.2032 | 4.3674 | 1,024 tokens |

### Key Findings

1. **Superior Performance**: EGPT achieves significantly lower training and validation losses compared to the baseline DeepGPT model

2. **Enhanced Context Understanding**: The extended context window allows EGPT to capture long-range dependencies and maintain coherence across longer sequences

3. **Scalable Architecture**: The hierarchical design enables efficient processing of very long sequences without quadratic scaling issues

4. **Transfer Learning Benefits**: Leveraging pre-trained GPT-2 encoder provides strong feature extraction capabilities

## Usage

### EGPT

```python
# replace config for train
from config import train_epgt_wap as config
from config.train_epgt_wap import *

# replace config for eval
from config import eval_epgt_wap as config
from config.eval_epgt_wap import *
```

```bash
python train_egpt.py
```

### Baseline

```python
# replace config for train
from config import train_dpgt_wap as config
from config.train_dpgt_wap import *

# replace config for eval
from config import eval_dpgt_wap as config
from config.eval_dpgt_wap import *
```

```bash
python train_dgpt.py
```

## Technical Details

### Memory Efficiency

- **Frozen Encoder**: Encoder parameters don't require gradients, reducing memory usage
- **Chunk Processing**: Sequential processing of chunks prevents memory overflow
- **Feature Caching**: Previous chunk features are cached for efficiency

### Computational Complexity

- **Time Complexity**: O(N + K×M) where N is total sequence length, K is number of chunks, M is chunk size
- **Space Complexity**: Linear with respect to sequence length (vs quadratic for standard attention)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

