# My GPT

A clean, educational implementation of a GPT (Generative Pre-trained Transformer) model in PyTorch, closely following the architecture of GPT-2.

## Features

- **From Scratch Implementation**: Includes `CausalSelfAttention`, `MLP`, `Block`, and the full `GPT` model class.
- **Training Pipeline**: Train on FineWeb-Edu dataset with gradient accumulation, mixed precision, and MPS/CUDA support.
- **Text Generation**: Inference script to generate text from a prompt.
- **Educational Resources**: Includes detailed notebooks and reference papers.

## Project Structure

```
├── model/
│   ├── gpt.py          # Model architecture (Attention, MLP, Transformer Blocks)
│   └── config.py       # Configuration dataclass
├── training_data/
│   ├── download_training_data.py   # Download FineWeb-Edu dataset
│   └── training_data_eda.py        # Dataset analysis and statistics
├── train.py            # Training script
├── main.py             # Inference script
├── detailed_descriptions/          # Educational Jupyter notebooks
└── papers/             # Original research papers
```

## Installation

This project requires Python 3.11+.

```bash
uv sync
```



## Quick Start

### 1. Download Training Data

Downloads the FineWeb-Edu dataset (~400MB parquet file):

```bash
uv run training_data/download_training_data.py
```

### 2. Explore the Data (Optional)

Analyze the dataset with statistics and sample outputs:

```bash
uv run training_data/training_data_eda.py
```

For cuda installed Win setups, delete the default torch library and install a cuda enabled one.
Check cuda version before installation, below command is meant to be an example and it's for cuda12.8. 

```bash
uv remove torch
uv add torch --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

This shows:
- Document count and file size
- Text length statistics (characters and tokens)
- Sample documents
- Token distribution percentiles

### 3. Train the Model

```bash
uv run train.py
```

Training features:
- **Hardware**: Auto-detects CUDA, MPS (Apple Silicon), or CPU
- **Gradient Accumulation**: Simulates large batch sizes on limited hardware
- **Mixed Precision**: Uses bfloat16/float16 for faster training
- **Sample Generation**: Prints generated text every 100 steps to visualize learning
- **Checkpoints**: Saves to `out_fineweb/ckpt.pt` when validation improves

Key hyperparameters (edit in `train.py`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 12 | Reduce if OOM errors |
| `gradient_accumulation_steps` | 40 | Effective batch = 480 |
| `block_size` | 1024 | Context window |
| `eval_interval` | 500 | Steps between evaluations |
| `sample_interval` | 100 | Steps between text samples |

### 4. Run Inference

Generate text with trained weights:

```bash
uv run main.py
```

Expects weights at `weights/trained_weights.pth`.

## References

- **Attention Is All You Need** (Vaswani et al., 2017)
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
