# PGPO Evaluation Guide

This directory contains the complete evaluation pipeline for PGPO. It measures generation quality, downstream recommendation utility, embedding-level privacy, and white-box source-variant unlinkability.

## Evaluation Overview

| Stage | Directory | Purpose |
|---|---|---|
| Item embedding construction | `embedding/` | Build semantic-baseline and PGPO-obfuscated item embeddings |
| Variant quality | `quality/` | Measure validity, diversity, semantic structure, fluency, NLL, and KL divergence |
| Recommendation | `recsys/` | Evaluate RecBole, multimodal, TALLRec, and LLaRA backends |
| Embedding inversion | `attack/` | Recover sensitive category labels from released item embeddings |
| White-box re-identification | `w2w_attack/` | Rank candidate source words using Replay-MLE-NoHistory |

## Prerequisites

Complete the preprocessing and PGPO generation steps in the [main README](../README.md) before running evaluation. In particular, the following files should be available:

```text
data/<dataset>/handled/id_item.json
data/<dataset>/handled/cleaned_item_edges.json
output/generated_variants_movielens.json
output/generated_variants_amazon-book.json
```

The generated-variant JSON file is expected to contain records with at least:

```json
{
  "original_word": "example source term",
  "variant": "generated private term",
  "category": "semantic category"
}
```

Unless stated otherwise, each command block below starts from the repository root.

## Version Convention

The evaluation scripts use version names to distinguish text and embedding sources:

| Version | Meaning |
|---|---|
| `v0` | No-text or lower-bound configuration where supported |
| `v2` | Semantic description baseline constructed from cleaned item edges |
| `v3` | PGPO-obfuscated description constructed from generated variants |
| `v3_custom` | Example name for a user-supplied variant file |

Custom version names are supported by most pipelines as long as the corresponding embedding directory exists.

## 1. Item Embedding Construction

The embedding pipeline creates the item representations consumed by quality, recommendation, and inversion evaluation.

Each generated version directory contains:

```text
item_embeddings.npy
item_id_map.json
item_content.json
```

### Semantic Baseline (`v2`)

```bash
cd evaluate/embedding

python generate_item_embeddings_v2.py --dataset ml-1m
python generate_item_embeddings_v2.py --dataset amazon-book
```

Outputs are written under `evaluate/embedding/data/<dataset>/v2/`.

### PGPO Variants (`v3`)

```bash
cd evaluate/embedding

python generate_item_embeddings_v3.py \
  --dataset ml-1m \
  --input ../../output/generated_variants_movielens.json

python generate_item_embeddings_v3.py \
  --dataset amazon-book \
  --input ../../output/generated_variants_amazon-book.json
```

To evaluate a custom generated-variant file:

```bash
cd evaluate/embedding

python generate_item_embeddings_v3.py \
  --dataset ml-1m \
  --input ../../output/generated_variants_movielens.json \
  --version_name v3_custom
```

## 2. Obfuscated Variant Quality

### Word-Level Quality

`test_word.py` reports lexical validity and diversity statistics. `evaluate_grpo_variants.py` additionally evaluates semantic displacement and category-aware neighborhood structure.

```bash
cd evaluate/quality
mkdir -p reports

python test_word.py \
  --generated_file ../../output/generated_variants_movielens.json \
  --output_file ./reports/ml1m_word_quality.json

python evaluate_grpo_variants.py \
  --generated_file ../../output/generated_variants_movielens.json \
  --output_file ./reports/ml1m_variant_structure.json
```

### Sentence-Level Quality

Evaluate fluency and diversity after inserting the generated variants into item descriptions:

```bash
cd evaluate/quality

python test_sentence.py \
  --input_file ../embedding/data/ml-1m/v3/item_content.json \
  --output_file ./reports/ml1m_v3_sentence_quality.json
```

Compare the token distributions of the obfuscated descriptions against the `v2` semantic baseline:

```bash
cd evaluate/quality

python calculate_sentence_kl_vs_v2.py \
  --dataset ml-1m \
  v3
```

Use `--data_root`, `--baseline_name`, and `--output_path` to override the default layout.

## 3. Recommendation Performance

Recommendation evaluation is organized into four backends:

```text
recsys/
├── multi/       # FREEDOM/BPR and LightGT
├── recbole/     # SASRec, LightGCN, and other RecBole models
├── tallrec/     # TALLRec-style LLM recommendation
└── llara/       # LLaRA-style semantic recommendation
```

Run each embedding variant separately to compare the semantic baseline (`v2`) with PGPO (`v3`).

### 3.1 Multimodal Backends: FREEDOM and LightGT

Prepare chronological interaction splits:

```bash
cd evaluate/recsys/multi

python preprocess.py --dataset ml-1m
python preprocess.py --dataset amazon-book
```

Run FREEDOM/BPR with a selected embedding version:

```bash
cd evaluate/recsys/multi

python run_mmrec_freedom.py \
  --dataset ml-1m \
  --edge_variant v3 \
  --embedding_root ../../embedding/data
```

Run LightGT:

```bash
cd evaluate/recsys/multi

python run_lightgt.py \
  --dataset ml-1m \
  --edge_variant v3 \
  --embedding_root ../../../embedding/data
```

The runner scripts perform hyperparameter search by default. Use `--max_trials` for smoke tests or limited-budget runs.

### 3.2 RecBole Backends: SASRec and LightGCN

Prepare RecBole-format interactions:

```bash
cd evaluate/recsys/recbole

python preprocess_recbole_data.py --dataset ml-1m
python preprocess_recbole_data.py --dataset amazon-book
```

Train and evaluate SASRec:

```bash
cd evaluate/recsys/recbole

python train_eval_recbole.py \
  --model sasrec \
  --dataset ml1m_recbole \
  --embedding_variant v3 \
  --data_path ./data/recbole \
  --embedding_root ../../embedding
```

For LightGCN, replace `--model sasrec` with `--model lightgcn`. The same entry point also supports `two_tower`, `deepfm`, and `neumf`.

To evaluate several models or embedding versions in sequence:

```bash
cd evaluate/recsys/recbole

python run_all_variants.py \
  --dataset ml1m_recbole \
  --models lightgcn,sasrec \
  --embedding_variants v2,v3
```

### 3.3 TALLRec

Prepare instruction-style train, validation, and test files:

```bash
cd evaluate/recsys/tallrec

python preprocess.py --dataset ml-1m
```

Train TALLRec with PGPO item embeddings:

```bash
cd evaluate/recsys/tallrec

python train.py \
  --base_model ../../../base_models/Qwen2.5-7B-Instruct \
  --dataset_name ml-1m \
  --embedding_version v3 \
  --embedding_root ../../embedding/data \
  --output_dir ./output/ml-1m_v3
```

Evaluate the trained checkpoint:

```bash
cd evaluate/recsys/tallrec

python evaluate.py \
  --base_model ../../../base_models/Qwen2.5-7B-Instruct \
  --dataset_name ml-1m \
  --embedding_version v3 \
  --embedding_root ../../embedding/data \
  --lora_weights ./output/ml-1m_v3
```

### 3.4 LLaRA

Prepare sequential recommendation data aligned with the selected embedding version:

```bash
cd evaluate/recsys/llara

python preprocess_llara_data.py \
  --dataset ml-1m \
  --embedding_variant v3
```

Train and test the LLaRA pipeline:

```bash
cd evaluate/recsys/llara

bash train_semantic.sh v3 generate ml-1m /path/to/your/llm
bash test_semantic.sh  v3 generate ml-1m /path/to/your/llm
```

The four positional shell-script arguments are:

```text
<embedding_variant> <task_type> <dataset_name> <llm_path>
```

## 4. Embedding Inversion Attack

The `attack/` pipeline evaluates whether an attacker can recover sensitive semantic categories from released item embeddings. It supports a locally trained T5-based inversion model and the official Vec2Text API.

The attack scripts use `movie` and `book` as dataset identifiers, rather than `ml-1m` and `amazon-book`.

### Step 1: Build the Attack Dataset

If the raw attack data is already available under `evaluate/attack/data/<dataset>/raw/`, convert it into the processed `[id, item_category]` representation:

```bash
cd evaluate/attack

python create_dataset.py --dataset movie
python create_dataset.py --dataset book
```

To construct the raw attack dataset itself, use the scripts under `attack/InvInst_dataset/`.

### Step 2: Build Attack Inputs

Generate training embeddings from the raw attack text:

```bash
cd evaluate/attack

python process_data.py \
  --dataset movie \
  --target_split train \
  --embedding_type v3 \
  --train_model_name_or_path ../../base_models/bert-base-uncased
```

Load the released PGPO embeddings for the test split:

```bash
cd evaluate/attack

python process_data.py \
  --dataset movie \
  --target_split test \
  --embedding_type v3 \
  --embedding_root ../embedding/data/ml-1m
```

### Step 3: Train the Local Inversion Model

```bash
cd evaluate/attack

python train_vector2text.py \
  --dataset movie \
  --stage all \
  --split train \
  --version v3 \
  --model_name_or_path ../../base_models/t5-base
```

### Step 4: Evaluate Inversion Robustness

Evaluate the locally trained model:

```bash
cd evaluate/attack

python evaluate.py \
  --dataset movie \
  --split test \
  --version v3 \
  --eval_mode local \
  --save_predictions
```

Alternatively, use the official Vec2Text corrector:

```bash
cd evaluate/attack

python evaluate.py \
  --dataset movie \
  --split test \
  --version v3 \
  --eval_mode vec2text
```

The reported metrics include exact match, edit similarity, label-overlap precision/recall/F1, and label-set Jaccard similarity.

## 5. Replay-MLE White-Box Re-identification

`w2w_attack/replay_mle_whitebox_attack.py` implements the Replay-MLE-NoHistory attack. The attacker knows the optimized conversion model, vocabulary, category metadata, and prompt template; it observes a released variant but does not observe the hidden generation history.

For every candidate source word, the script replays the known prompt with an empty history and computes the teacher-forced log-likelihood of the observed variant. Candidates are ranked by this score.

### Sampled Same-Category Candidate Set

The following example evaluates each ground-truth source against 50 randomly sampled same-category negatives:

```bash
python evaluate/w2w_attack/replay_mle_whitebox_attack.py \
  --pairs_json_path ./output/generated_variants_movielens.json \
  --lora_path ./output/grpo_model/movielens \
  --prompt_file ./models/prompt/prompt.txt \
  --candidate_mode sampled_same_category \
  --num_random_negatives 50 \
  --seed 2026 \
  --output_path ./output/w2w_replay_mle_movielens.json \
  --save_details
```

For Amazon-Book, replace the input, LoRA, and output paths with their `amazon-book` counterparts.

### Full-Vocabulary Candidate Set

For exhaustive closed-vocabulary evaluation:

```bash
python evaluate/w2w_attack/replay_mle_whitebox_attack.py \
  --pairs_json_path ./output/generated_variants_movielens.json \
  --lora_path ./output/grpo_model/movielens \
  --candidate_mode all \
  --output_path ./output/w2w_replay_mle_movielens_all.json
```

This mode can be substantially more expensive because every observed variant is scored against the complete source vocabulary.

### Reported Metrics

- `ReID@K`: fraction of examples whose true source is ranked within the top `K` candidates;
- `MRR`: mean reciprocal rank of the true source;
- `posterior_entropy_bits`: entropy of the attack-induced posterior;
- `mi_proxy_bits`: `log2(candidate_count) - posterior_entropy_bits`;
- `effective_anonymity_set`: exponential of the posterior entropy;
- `true_source_nll_bits`: negative log posterior assigned to the true source.

Use `--rank_eval_ks`, `--posterior_temperature`, and `--sampling_trials` to customize ranking, posterior calibration, and repeated negative sampling.

## Reproducibility Notes

- Keep the interaction splits and random seed fixed when comparing `v2` and `v3`.
- Use the same embedding encoder and dimensionality across methods.
- Compare privacy attacks under matched candidate-set construction.
- Record all non-default command-line arguments together with the generated JSON reports.
- The Replay-MLE entropy and mutual-information values are attack-induced operational proxies under the evaluated candidate set; they are not full-vocabulary information-theoretic guarantees.

## Adding a Custom Variant

To evaluate another obfuscation method:

1. store its source-variant mappings in the same JSON format as the PGPO output;
2. generate a new embedding version with `generate_item_embeddings_v3.py --version_name <name>`;
3. run quality and recommendation evaluation with the same data splits and encoder;
4. run both embedding inversion and Replay-MLE with matched attack settings.
