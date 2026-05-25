import argparse
import ast
import gzip
import json
import os
from tqdm import tqdm

DATASET_CONFIGS = {
    'ml-1m': {
        'input_file': './data/ml-1m/raw/movies.dat',
        'output_file': './data/ml-1m/handled/id_item.json'
    },
    'amazon-book': {
        'input_file': './data/amazon-book/raw-0/meta_Books.json.gz',
        'output_file': './data/amazon-book/handled/id_item.json'
    }
}

def detect_file_format(file_path):
    lower_path = file_path.lower()
    if lower_path.endswith('.json'):
        return 'json'
    elif lower_path.endswith('.json.gz') or 'meta_books' in lower_path or 'meta_' in lower_path:
        return 'amazon-meta-gz'
    elif 'movies.dat' in lower_path:
        return 'ml-1m'
    elif 'u.item' in lower_path or lower_path.endswith('.item'):
        return 'movielens'
    return 'unknown'

def parse_json_line(line):
    business = json.loads(line.strip())
    business_id = business.get('business_id')
    name = business.get('name')
    return business_id, name

def parse_movielens_line(line):
    parts = line.strip().split('|')
    if len(parts) >= 2:
        item_id = parts[0]
        name = parts[1]
        return item_id, name
    return None, None

def parse_ml1m_line(line):
    parts = line.strip().split('::')
    if len(parts) >= 2:
        item_id = parts[0]
        name = parts[1]
        return item_id, name
    return None, None

def parse_amazon_meta_line(line):
    line = line.strip()
    if not line:
        return None, None
    try:
        book = json.loads(line)
    except json.JSONDecodeError:
        try:
            book = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None, None
    item_id = book.get('asin')
    name = book.get('title') or book.get('name')
    if isinstance(name, str):
        name = name.strip()
    if not name:
        name = item_id
    return item_id, name

def get_line_iterator(input_file, file_format):
    if file_format == 'amazon-meta-gz':
        return gzip.open(input_file, 'rt', encoding='utf-8', errors='ignore')
    return open(input_file, 'r', encoding='utf-8', errors='ignore')

def extract_id_name(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    id_name_map = {}
    
    print(f"Starting to read file: {input_file}")
    
    file_format = detect_file_format(input_file)
    print(f"Detected file format: {file_format}")
    if file_format == 'unknown':
        raise ValueError(f"Unsupported file format for input file: {input_file}")
    
    print("Calculating total number of lines in file...")
    total_lines = 0
    with get_line_iterator(input_file, file_format) as f:
        for _ in f:
            total_lines += 1
    print(f"Total number of lines in the document: {total_lines}")
    
    print("Beginning to extract the mapping from ID to item name...")
    line_number = 0
    with get_line_iterator(input_file, file_format) as f:
        for line_number, line in tqdm(enumerate(f, 1), total=total_lines, desc="Processing"):
            try:
                if file_format == 'json':
                    item_id, name = parse_json_line(line)
                elif file_format == 'movielens':
                    item_id, name = parse_movielens_line(line)
                elif file_format == 'ml-1m':
                    item_id, name = parse_ml1m_line(line)
                elif file_format == 'amazon-meta-gz':
                    item_id, name = parse_amazon_meta_line(line)
                else:
                    raise ValueError(f"Unknown file format: {file_format}")
                
                if item_id and name:
                    id_name_map[item_id] = name
                    
            except json.JSONDecodeError as e:
                print(f"Parsing error on line {line_number}: {e}")
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
    
    print(f"File reading completed, processed {line_number} lines")
    print(f"Generated mapping count: {len(id_name_map)}")
    
    print(f"Start saving mapped to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(id_name_map, f, ensure_ascii=False, indent=2)
    
    print(f"Mapping saved successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='?', default=None)
    parser.add_argument('output_file', nargs='?', default=None)
    parser.add_argument(
        '--dataset',
        choices=list(DATASET_CONFIGS.keys()),
        default='ml-1m'
    )
    args = parser.parse_args()
    defaults = DATASET_CONFIGS[args.dataset]
    input_file = args.input_file or defaults['input_file']
    output_file = args.output_file or defaults['output_file']
    extract_id_name(input_file, output_file)
