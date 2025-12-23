import json
from pathlib import Path

from agentless.multilang.const import LANGUAGE, LANG_EXT


def process(raw_data):
    raw = json.loads(raw_data)
    # MOBILE_DEV: Handle None values in title and body
    title = raw['resolved_issues'][0]['title']
    body = raw['resolved_issues'][0]['body']

    if title is None or body is None or title == '' or body == '':
        print(f"Warning: instance_id {raw['instance_id']} has empty title or body - title: {repr(title)}, body: {repr(body)}")

    title = title or ''
    body = body or ''
    data = {
        'repo': f'{raw["org"]}/{raw["repo"]}',
        'instance_id': raw['instance_id'],
        'base_commit': raw['base']['sha'],
        'problem_statement': title + '\n' + body,
    }
    return data


def load_local_json(language=None):
    """Load dataset from local JSON files in data/{language}/ directory."""
    if language is None:
        language = LANGUAGE

    # Simple mapping for folder names
    lang_folder_map = {
        'javascript': 'js',
        'typescript': 'ts',
    }
    folder_name = lang_folder_map.get(language, language)

    path = Path(f'data/{folder_name}')
    lines = []

    if path.exists():
        for file in path.iterdir():
            if file.is_file():
                lines.extend(file.read_text().splitlines())

    # MOBILE_DEV: Filter out None results from process (invalid JSON lines)
    dataset = [d for d in (process(x) for x in lines if x.strip()) if d is not None]
    return dataset


def end_with_ext(file_name):
    for ext in LANG_EXT:
        if file_name.endswith(f'.{ext}'):
            return True
    return False
