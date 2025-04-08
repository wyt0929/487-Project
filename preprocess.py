import os
import json
import spacy
from tqdm import tqdm
output_dir = 'processed_jsons'
os.makedirs(output_dir, exist_ok=True)
nlp = spacy.load("en_core_web_sm")
input_dir = 'jsons'
json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
for filename in tqdm(json_files, desc="Processing files"):
    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    content = data.get('content', '')
    doc = nlp(content)
    filtered_words = [token.text for token in doc if token.pos_ in {'VERB', 'ADJ', 'ADV'}]
    processed_content = ' '.join(filtered_words)
    data['content'] = processed_content
    output_filepath = os.path.join(output_dir, filename)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)