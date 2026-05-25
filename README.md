# Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation

PGPO is a prototype-guided privacy-preserving framework for LLM-enhanced recommendation. It builds item-level semantic structures, prunes noisy edges, constructs prototype representations, and then optimizes obfuscation quality with GRPO-based reward learning.

## Overview

This repository provides:

- Data preprocessing pipelines for `MovieLens-1M` and `Amazon-Book`
- Semantic edge construction from item titles and metadata
- Prototype extraction and similarity computation for fine-grained details
- GRPO training for privacy-preserving variant generation

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://anonymous.4open.science/r/PGPO/
cd PGPO
```

### 2. Create the Python Environment

```bash
conda create --prefix ./PGPO python=3.10 -y
conda activate ./PGPO
pip install -r requirements.txt
```

### 3. Download Base Models

```bash
mkdir -p base_models
python download_base_models.py
```

## Data Preparation

### MovieLens-1M

- Download `ml-1m.zip` from [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)
- Extract files such as `ratings.dat`, `movies.dat`, and `users.dat`
- Place them under `data/ml-1m/raw/`

### Amazon-Book

- Download the interaction file [Books.csv](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books.csv)
- Download the metadata file [meta_Books.json.gz](https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Books.json.gz)
- Place both files under `data/amazon-book/raw/`

### Expected Directory Layout

```text
PGPO/
├── base_models/
│   ├── Qwen3-4B-Base/
│   └── ...
├── data/
│   ├── ml-1m/
│   │   └── raw/
│   └── amazon-book/
│       └── raw/
├── download_base_models.py
├── extract_id_name.py
├── prepare_amazon_raw0.py
├── compute_embeddings.py
└── ...
```

## Preprocessing Pipeline

Most preprocessing scripts support two usage patterns:

- Default-path mode: only specify `--dataset`, and the script uses built-in paths
- Explicit-path mode: directly pass input and output paths as positional arguments

### A. Amazon-Book Subset Construction

Use the default preprocessing configuration:

```bash
python prepare_amazon_raw0.py --dataset amazon-book
```

A more controllable example:

```bash
python prepare_amazon_raw0.py \
  --dataset amazon-book \
  --mode window_freq_sample \
  --start_date 2018-01-01 \
  --end_date 2018-03-31 \
  --target_items 10000 \
  --min_item_freq 3 \
  --min_user_freq 3 \
  --seed 42
```

This script reads raw Amazon data and produces a filtered subset under `data/amazon-book/raw-0/`.

### B. MovieLens-1M Raw Copy

For the MovieLens pipeline, duplicate the raw directory as the initial processed snapshot:

```bash
cp -r data/ml-1m/raw data/ml-1m/raw-0
```

### C. Extract Item ID-Name Mapping

Default usage for `MovieLens-1M`:

```bash
python extract_id_name.py --dataset ml-1m
```

Default usage for `Amazon-Book`:

```bash
python extract_id_name.py --dataset amazon-book
```

Explicit input and output paths:

```bash
python extract_id_name.py \
  ./data/ml-1m/raw/movies.dat \
  ./data/ml-1m/handled/id_item.json
```

This step generates an `id -> item_name` mapping file, which is the entry point for downstream semantic edge construction.

### D. Generate Semantic Edges

Run with dataset defaults:

```bash
python generate_edges/generate_edges_local.py --dataset ml-1m
```

Or provide custom arguments explicitly:

```bash
python generate_edges/generate_edges_local.py \
  ./data/ml-1m/handled/id_item.json \
  ./data/ml-1m/handled/item_edges.json \
  ./base_models/Qwen2.5-14B-Instruct
```

This stage invokes the LLM to generate fine-grained semantic attributes for each item.

### E. Prune Noisy Edges

```bash
python cut_item_edges.py --dataset ml-1m
```

Or:

```bash
python cut_item_edges.py \
  ./data/ml-1m/handled/item_edges.json \
  ./data/ml-1m/handled/cleaned_item_edges.json
```

This step removes low-quality or redundant category edges and retains a more stable semantic structure.

### F. Compute Prototype Embeddings and Similarities

Run with default paths:

```bash
python compute_embeddings.py --dataset ml-1m
```

Or specify all major inputs explicitly:

```bash
python compute_embeddings.py \
  ./data/ml-1m/handled/cleaned_item_edges.json \
  ./data/ml-1m/handled/extracted \
  ./data/ml-1m/handled/extracted/grpo_dataset \
  ./base_models/bert-base-uncased \
  512 \
  10 \
  10000
```

This stage produces:

- extracted fine-grained item details
- embedding representations for those details
- similarity statistics used by GRPO training

## Recommended End-to-End Commands

### MovieLens-1M

```bash
cp -r data/ml-1m/raw data/ml-1m/raw-0
python extract_id_name.py --dataset ml-1m
python generate_edges/generate_edges_local.py --dataset ml-1m
python cut_item_edges.py --dataset ml-1m
python compute_embeddings.py --dataset ml-1m
```

### Amazon-Book

```bash
python prepare_amazon_raw0.py --dataset amazon-book
python extract_id_name.py --dataset amazon-book
python generate_edges/generate_edges_local.py --dataset amazon-book
python cut_item_edges.py --dataset amazon-book
python compute_embeddings.py --dataset amazon-book
```

## GRPO Training

After preprocessing is complete, you can launch GRPO training with dataset-aware default paths:

```bash
python train_grpo.py --dataset movielens
```

For Amazon-Book:

```bash
python train_grpo.py --dataset amazon-book
```

A more explicit example:

```bash
python train_grpo.py \
  --dataset movielens \
  --model_name_or_path ./base_models/Qwen3-4B-Base \
  --learning_rate 5e-7 \
  --batch_size 2 \
  --max_steps 10000 \
  --num_generations 8 \
  --use_cache 1
```

downstream evaluation
embedding
cd evaluate/embedding
python generate_item_embeddings_v2.py
python generate_item_embeddings_v3.py 

Obfuscated Variants Quality
cd evaluate/quality

Recommendation Performance
cd recsys

Embedding Inversion Defense
cd attack