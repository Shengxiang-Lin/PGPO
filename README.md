<h1 align="center">Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation</h1>

## Overview

This repository covers:

- Data preprocessing pipelines for `MovieLens-1M` and `Amazon-Book`
- LLM-based semantic edge construction and category-aware edge pruning
- Prototype extraction and similarity computation for fine-grained item details
- GRPO training for privacy-preserving variant generation
- Downstream evaluation for embedding quality, recommendation performance, and inversion robustness
- Reproducible experimental entry points for both semantic baselines and obfuscated variants

### Workflow at a Glance

1. Prepare raw datasets and construct cleaned semantic item structures.
2. Build prototype-level embeddings and similarity statistics for GRPO training.
3. Train PGPO to generate privacy-preserving obfuscated variants.
4. Evaluate the resulting variants from embedding, quality, recommendation, and defense perspectives.

## Environment Setup

This section prepares the runtime environment, base models, and raw datasets required by the preprocessing and evaluation pipelines.

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

This section converts raw datasets into the structured semantic resources required by PGPO. Most preprocessing scripts support two usage patterns:

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

## Quick Start Pipelines

The following commands summarize the recommended preprocessing flow for each dataset.

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

Once the preprocessing stage is complete, you can train PGPO to generate privacy-preserving variants with dataset-aware default paths.

Launch training for MovieLens:

```bash
python train_grpo.py --dataset movielens
```

Launch training for Amazon-Book:

```bash
python train_grpo.py --dataset amazon-book
```

A more explicit configuration example:

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

## Downstream Evaluation

All downstream evaluation code is located under `evaluate/`. This part measures the utility, quality, and privacy properties of the generated variants. Before running this section, make sure that:

- the preprocessing pipeline has finished successfully
- `cleaned_item_edges.json` and `id_item.json` are available under `data/<dataset>/handled/`
- GRPO-generated variants are available if you want to evaluate obfuscated text embeddings (`v3`)

### 1. Embedding Construction

This stage converts item text into dense item embeddings for downstream tasks. The generated files are saved to:

- `evaluate/embedding/data/<dataset>/v2/`
- `evaluate/embedding/data/<dataset>/v3/`

Each variant directory contains:

- `item_embeddings.npy`
- `item_id_map.json`
- `item_content.json`

Generate the semantic baseline embedding (`v2`):

```bash
cd evaluate/embedding
python generate_item_embeddings_v2.py --dataset ml-1m
python generate_item_embeddings_v2.py --dataset amazon-book
```

Generate the obfuscated embedding (`v3`) from GRPO variants:

```bash
cd evaluate/embedding
python generate_item_embeddings_v3.py --dataset ml-1m
python generate_item_embeddings_v3.py --dataset amazon-book
```

If you want to evaluate a custom variant file, `generate_item_embeddings_v3.py` supports explicit input override:

```bash
python generate_item_embeddings_v3.py \
  --dataset ml-1m \
  --input /path/to/generated_variants.json \
  --version_name v3_custom
```

### 2. Obfuscated Variants Quality

This part evaluates generated variants from lexical, semantic, and distributional perspectives. The relevant scripts are under `evaluate/quality/`.

```bash
cd evaluate/quality
```

Word-level quality evaluation:

```bash
python test_word.py \
  --generated_file /path/to/generated_variants.json \
  --output_file ./reports/word_quality.json
```

Semantic preservation and category-structure evaluation:

```bash
python evaluate_grpo_variants.py \
  --generated_file /path/to/generated_variants.json \
  --output_file ./reports/grpo_variant_eval.json
```

Sentence-level fluency, diversity, and coherence evaluation for item texts:

```bash
python test_sentence.py \
  --input_file ../embedding/data/ml-1m/v3/item_content.json \
  --output_file ./reports/v3_sentence_quality.json
```

KL divergence against the `v2` semantic baseline:

```bash
python calculate_sentence_kl_vs_v2.py \
  --dataset ml-1m \
  v3
```

Notes:

- `test_word.py` and `evaluate_grpo_variants.py` expect a JSON array with fields such as `original_word`, `variant`, and `category`
- `test_sentence.py` accepts either a JSON object or a JSON list of texts
- `calculate_sentence_kl_vs_v2.py` compares `item_content.json` in the target variant directory against the `v2` baseline under `../embedding/data/<dataset>/`

### 3. Recommendation Performance

Recommendation evaluation is implemented under `evaluate/recsys/` with three families of backends:

- `multi/`: multimodal baselines such as MMRec FREEDOM and LightGT
- `recbole/`: RecBole-based recommenders such as SASRec, LightGCN, NeuMF, DeepFM, and Two-Tower variants
- `llara/`: LLaRA-style semantic recommendation pipeline

#### Multi-Modal Baselines

Prepare interaction splits for multimodal recommenders:

```bash
cd evaluate/recsys/multi
python preprocess.py --dataset ml-1m
python preprocess.py --dataset amazon-book
```

Run MMRec FREEDOM/BPR with semantic item embeddings:

```bash
python run_mmrec_freedom.py \
  --dataset ml-1m \
  --edge_variant v2 \
  --embedding_root ../../embedding/data
```

Run LightGT:

```bash
python run_lightgt.py \
  --dataset ml-1m \
  --edge_variant v2 \
  --embedding_root ../../../embedding/data
```

Notes:

- `run_mmrec_freedom.py` uses `v0` as the no-text baseline and `v2`/`v3` as text-enhanced variants
- `run_lightgt.py` forwards `edge_variant` to `LightGT/main.py`, so `v2` and `v3` can be compared directly

#### RecBole Backends

Prepare RecBole-format data:

```bash
cd ../recbole
python preprocess_recbole_data.py --dataset ml-1m
python preprocess_recbole_data.py --dataset amazon-book
```

Train and evaluate a RecBole model with pretrained item embeddings:

```bash
python train_eval_recbole.py \
  --model sasrec \
  --dataset ml1m_recbole \
  --embedding_variant v2 \
  --data_path ./data/recbole \
  --embedding_root ../../embedding
```

You can replace `sasrec` with `two_tower`, `deepfm`, `neumf`, or `lightgcn`.

#### LLaRA Evaluation

Prepare sequence data aligned with a chosen embedding variant:

```bash
cd ../llara
python preprocess_llara_data.py --dataset ml-1m --embedding_variant v2
python preprocess_llara_data.py --dataset amazon-book --embedding_variant v2
```

Train and test the LLaRA pipeline:

```bash
bash train_semantic.sh v2 generate ml-1m /path/to/your/llm
bash test_semantic.sh v2 generate ml-1m /path/to/your/llm
```

Notes:

- `preprocess_llara_data.py` reads embeddings from `evaluate/embedding/data/<dataset>/<variant>/`
- the shell scripts default to an external model path, so in practice you should pass your own local LLM path explicitly

### 4. Embedding Inversion Defense

The embedding inversion defense pipeline is implemented under `evaluate/attack/`. Its goal is to measure how difficult it is to recover category or semantic information from released item embeddings.

```bash
cd evaluate/attack
```

The full pipeline contains four steps.

#### Step 1. Build Attack Dataset

If the raw attack data has already been prepared under `evaluate/attack/data/<dataset>/raw/`, convert it to the processed `[id, item_category]` format:

```bash
python create_dataset.py --dataset movie
python create_dataset.py --dataset book
```

Notes:

- here the dataset names are `movie` and `book`, not `ml-1m` and `amazon-book`
- if you still need to construct the raw attack dataset itself, see `evaluate/attack/InvInst_dataset/`

#### Step 2. Inject Embeddings

Generate attack inputs from the train split or load external embeddings for the test split:

```bash
python process_data.py \
  --dataset movie \
  --target_split train \
  --embedding_type v2 \
  --train_model_name_or_path ../../../base_models/bert-base-uncased
```

```bash
python process_data.py \
  --dataset movie \
  --target_split test \
  --embedding_type v2 \
  --embedding_root ../embedding/data/ml-1m
```

Notes:

- `train` split creates embeddings locally from raw text features
- `test` split loads embeddings from an external embedding directory such as `evaluate/embedding/data/ml-1m`
- `process_data.py` contains environment-specific defaults pointing to another workspace, so it is recommended to pass `--train_model_name_or_path` and `--embedding_root` explicitly

#### Step 3. Train the Inversion Model

Prepare vec2text-style training data and train the local inversion model:

```bash
python train_vector2text.py \
  --dataset movie \
  --stage all \
  --split train \
  --version v2 \
  --model_name_or_path /path/to/t5-base
```

This script first prepares `jsonl` training files and then trains a T5-based inversion model with a learned projector.

#### Step 4. Evaluate Inversion Robustness

Evaluate the trained inversion model on the processed embeddings:

```bash
python evaluate.py \
  --dataset movie \
  --split test \
  --version v2 \
  --eval_mode local \
  --save_predictions
```

If you want to evaluate with the official `vec2text` API instead of the local model:

```bash
python evaluate.py \
  --dataset movie \
  --split test \
  --version v2 \
  --eval_mode vec2text
```

The evaluation script reports metrics such as:

- exact match accuracy
- edit similarity
- label set overlap precision / recall / F1
- Jaccard similarity on recovered category labels
