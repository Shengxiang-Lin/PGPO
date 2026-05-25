import json
import os
from tqdm import tqdm
from openai import OpenAI

class BusinessEdgeGenerator:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com"):
        if api_key is None:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key is None:
                api_key = "sk-c44ee37fba784173929297586940a07d"
                print("Warning: Using default API key. Please set DEEPSEEK_API_KEY environment variable for production use.")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def load_businesses(self, id_item_path='./data/yelp/handled/id_item_failed.json'):
        with open(id_item_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_business_edge(self, business_name):
        prompt_template = '''You are a professional knowledge graph and recommendation systems expert. Your task is to generate a two-layer tree-structured side information for the given business name.

CRITICAL REQUIREMENTS:
- Output MUST be EXACTLY and ONLY the valid JSON object. No explanations, no extra text, no markdown, no "Here is the output", no trailing text after the closing }}.
- The "business_name" field MUST be exactly: "{business_name}" (copy it verbatim, including any hyphens, commas, or special characters).
- Do not add any commentary inside or outside the JSON.
- For each of the 3 items in "hierarchy", generate a concise and specific "category_name" in English that accurately represents the coarse-grained category for this business (e.g., "Food & Beverage" instead of generic "Industry Attributes").
- All "fine_details" must be in English, specific, distinguishable, and 2-4 items per category.

JSON format (output exactly this structure):
{{
  "business_name": "{business_name}",
  "hierarchy": [
    {{
      "relation": "belongs_to_industry",
      "category_name": "specific industry category name generated for this business",
      "sub_relation": "has_feature",
      "fine_details": ["specific industry feature 1", "specific industry feature 2", "specific industry feature 3"]
    }},
    {{
      "relation": "offers_service",
      "category_name": "specific core services category name generated for this business",
      "sub_relation": "has_feature",
      "fine_details": ["specific service 1", "specific service 2", "specific service 3", "specific service 4"]
    }},
    {{
      "relation": "targets_customer",
      "category_name": "specific target customers category name generated for this business",
      "sub_relation": "has_feature",
      "fine_details": ["customer group 1", "customer group 2", "customer group 3"]
    }}
  ]
}}

Business name: {business_name}

Output only the JSON now.'''
        
        prompt = prompt_template.format(business_name=business_name)

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a professional knowledge graph and recommendation systems expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            edge_info = response.choices[0].message.content.strip()
            
            try:
                edge_info_json = json.loads(edge_info)
                return edge_info_json
            except json.JSONDecodeError:
                try:
                    if not edge_info.startswith('{'):
                        start_idx = edge_info.find('{')
                        if start_idx != -1:
                            edge_info = edge_info[start_idx:]
                    
                    if not edge_info.endswith('}'):
                        end_idx = edge_info.rfind('}') + 1
                        if end_idx != -1:
                            edge_info = edge_info[:end_idx]
                    
                    edge_info_json = json.loads(edge_info)
                    return edge_info_json
                except Exception:
                    return None
                    
        except Exception:
            return None
    
    def generate_business_edges(self, businesses, output_path='./data/yelp/handled/failed_retry.json'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        business_edges = {}
        failed_items = []
        
        print(f"Begin generating edge information for the business name...")
        print(f"Edge information writes generated edges to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('{}')
        
        for business_id, business_name in tqdm(businesses.items(), desc="Generate edge information"):
            try:
                edge_info = self.generate_business_edge(business_name)
                
                if edge_info:
                    business_edges[business_id] = edge_info
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(business_edges, f, ensure_ascii=False, indent=2)
                else:
                    failed_items.append({"id": business_id, "name": business_name})
            except Exception:
                failed_items.append({"id": business_id, "name": business_name})
        
        print(f"Edge information generation and writing completed! A total of {len(business_edges)} items were successfully generated, with {len(failed_items)} failures.")
        if failed_items:
            print(f"Failed list length: {len(failed_items)}")
        
        return business_edges

def main():
    generator = BusinessEdgeGenerator()
    
    businesses = generator.load_businesses()
    print(f"A total of {len(businesses)} business names have been loaded.")
    
    test_businesses = dict(list(businesses.items())[:10])
    print(f"Test Mode: Process only the first 10 business names")
    
    results = generator.generate_business_edges(test_businesses)
    
    print(f"Successful generation of edge information: {len(results)} edges")
    
    return results


if __name__ == '__main__':
    main()
