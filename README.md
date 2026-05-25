# Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation

## Setup
### Clone the Repository
```bash
git clone https://anonymous.4open.science/r/PGPO/
cd PGPO
```
### Download Base Models
```bash
mkdir base_models
python download_base_models.py
```
### Download and Place the Datasets
**MovieLens-1M**
- Download `ml-1m.zip` from [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)
- Extract the contents (e.g., `ratings.dat`, `movies.dat`, `users.dat`) and place them into the `data/ml-1m/raw/` directory
**Amazon-Books**
- Download the ratings file:[Books.csv](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books.csv)
- Download the metadata file:[meta_Books.json.gz](https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Books.json.gz)
- Save both files to the `data/amazon-books/raw/` directory
### Verify the Directory Structure
After completing the steps above, your project structure should look like this:
```
PGPO/
├── base_models/         # Pre-trained base models
├── ├── Qwen3-4B-Base/
│   ├── Qwen2.5-14B-Instruct/
├── data/
│   ├── ml-1m/
│   │   └── raw/          # MovieLens-1M raw data
│   └── amazon-books/
│       └── raw/          # Books.csv and meta_Books.json.gz
├── download_base_models.py
└── ...
```

