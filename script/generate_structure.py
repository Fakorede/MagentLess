import json
import os
import sys

sys.path.append('.')

from pathlib import Path
from tqdm import tqdm

from get_repo_structure.get_repo_structure import get_project_structure_from_scratch

PLAYGROUND = 'playground'
OUT_DIR = 'structure'

# Get language from environment or default to java
LANGUAGE = os.environ.get('SWEBENCH_LANG', 'java')


def load_dataset(language):
    """Load dataset from data/{language}/ directory."""
    # Map language names to folder names
    lang_folder_map = {
        'javascript': 'js',
        'typescript': 'ts',
    }
    folder_name = lang_folder_map.get(language, language)
    
    path = Path(f'data/{folder_name}')
    dataset = []
    
    if path.exists():
        for file in path.iterdir():
            if file.is_file() and file.suffix == '.jsonl':
                for line in file.read_text().splitlines():
                    if line.strip():
                        raw = json.loads(line)
                        dataset.append({
                            'repo': f'{raw["org"]}/{raw["repo"]}',
                            'instance_id': raw['instance_id'],
                            'base_commit': raw['base']['sha'],
                        })
    
    return dataset


def main():
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    dataset = load_dataset(LANGUAGE)
    print(f"Found {len(dataset)} instances for language: {LANGUAGE}")
    
    for data in tqdm(dataset):
        output_file = f'{OUT_DIR}/{data["instance_id"]}.json'
        
        # Skip if already generated
        if os.path.exists(output_file):
            print(f"Skipping {data['instance_id']} - already exists")
            continue
        
        try:
            structure = get_project_structure_from_scratch(
                repo_name=data['repo'],
                commit_id=data['base_commit'],
                instance_id=data['instance_id'],
                repo_playground=PLAYGROUND,
            )
            with open(output_file, 'w') as fout:
                json.dump(structure, fout)
            print(f"Generated structure for {data['instance_id']}")
        except Exception as e:
            print(f"Error generating structure for {data['instance_id']}: {e}")


if __name__ == '__main__':
    main()
