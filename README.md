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

The diagram above illustrates the complete EGPT architecture flow:

1. **Input Processing**: Long sequences (16,384+ tokens) that exceed standard GPT context
2. **Sequence Chunking**: Division into encoder-sized chunks (1024 tokens each)
3. **Feature Extraction**: Processing through frozen GPT-2 encoder
4. **Feature Enhancement**: Adding positional and length embeddings
5. **Hierarchical Processing**: EGPT decoder layers process chunk relationships
6. **Output Generation**: Final language model head produces token predictions

## How It Works

### Context Extension Mechanism

1. **Input Chunking**: Input tokens are divided into chunks of size equal to the encoder's context window (e.g., 1024 tokens for GPT-2)

2. **Feature Extraction**: Each chunk is processed by the frozen pre-trained encoder to extract high-level contextual features

3. **Hierarchical Processing**: The EGPT decoder processes relationships between chunk features, enabling understanding of long-range dependencies

4. **Exponential Scaling**: The effective context length becomes `encoder_block_size × egpt_block_size` (e.g., 1024 × 16 = 16,384 tokens)

## Data Preparation

### Datasets

The model is evaluated on multiple long-context datasets to assess its ability to understand extended literary works:

#### Dataset 1: War and Peace (Single Work)
- **Content**: Complete text of Tolstoy's "War and Peace"
- **Training**: 80% of the dataset  
- **Validation**: 20% of the dataset

#### Dataset 2: Multi-Author Literary Corpus
- **Training Set**: Complete texts of Tolstoy's works:
  - "War and Peace" 
  - "Anna Karenina"
- **Validation Set**: Herman Melville's "Moby Dick"
- **Purpose**: Tests cross-author generalization and diverse literary style understanding

### Setup

Run the data preparation script:

```bash
python prepare.py
```

This script will:
- Download the literary datasets
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

#### Dataset 1: War and Peace (Single Work)

| Model | Train Loss | Validation Loss | Context Length |
|-------|------------|-----------------|----------------|
| **EGPT** | **2.1431** | **3.3893** | 16,384 tokens |
| DeepGPT | 3.2032 | 4.3674 | 1,024 tokens |

#### Dataset 2: Multi-Author Literary Corpus

| Model | Train Loss | Validation Loss | Context Length |
|-------|------------|-----------------|----------------|
| **EGPT** | **2.7405** | **4.8718** | 16,384 tokens |
| DeepGPT | 3.8281 | 5.8239 | 1,024 tokens |

### Key Findings

1. **Consistent Superior Performance**: EGPT achieves significantly lower training and validation losses compared to the baseline DeepGPT model across both datasets

2. **Cross-Author Generalization**: On the multi-author dataset, EGPT demonstrates better generalization from Tolstoy's works (training) to Melville's "Moby Dick" (validation), showing improved cross-author understanding

3. **Enhanced Context Understanding**: The extended context window allows EGPT to capture long-range dependencies and maintain coherence across longer literary sequences

4. **Scalable Architecture**: The hierarchical design enables efficient processing of very long sequences without quadratic scaling issues

5. **Transfer Learning Benefits**: Leveraging pre-trained GPT-2 encoder provides strong feature extraction capabilities for diverse literary styles

6. **Performance Analysis**: 
   - **Single Author**: EGPT shows 33.1% improvement in train loss and 22.4% improvement in validation loss
   - **Multi-Author**: EGPT shows 28.4% improvement in train loss and 16.3% improvement in validation loss, demonstrating robust generalization capabilities

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

