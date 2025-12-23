import os

from agentless.multilang.example import (
    DIFF_C,
    DIFF_CPP,
    DIFF_DART,
    DIFF_GO,
    DIFF_JAVA,
    DIFF_JAVASCRIPT,
    DIFF_KOTLIN,
    DIFF_PYTHON,
    DIFF_RUST,
    DIFF_TYPESCRIPT,
)


def get_config(language):
    """
    Get configuration for a language.
    
    Single-language configs process files of one language.
    Multi-language configs (android, react_native, flutter) process files
    of multiple languages - useful for mobile repos that mix languages.
    """
    configs = {
        # Single language configs
        'python': {
            'LANG_EXT': ['py'],
            'DIFF_EXAMPLE': DIFF_PYTHON,
        },
        'java': {
            'LANG_EXT': ['java', 'kt', 'kts', 'xml', 'gradle'],
            'DIFF_EXAMPLE': DIFF_JAVA,
        },
        'kotlin': {
            'LANG_EXT': ['kt', 'kts', 'java', 'xml', 'gradle'],
            'DIFF_EXAMPLE': DIFF_KOTLIN,
        },
        'dart': {
            'LANG_EXT': ['dart', 'java', 'kt', 'xml', 'gradle', 'yaml'],
            'DIFF_EXAMPLE': DIFF_DART,
        },
        'go': {
            'LANG_EXT': ['go'],
            'DIFF_EXAMPLE': DIFF_GO,
        },
        'rust': {
            'LANG_EXT': ['rs'],
            'DIFF_EXAMPLE': DIFF_RUST,
        },
        'cpp': {
            'LANG_EXT': ['cpp', 'cxx', 'cc', 'c', 'hpp', 'hxx', 'h'],
            'DIFF_EXAMPLE': DIFF_CPP,
        },
        'c': {
            'LANG_EXT': ['c', 'h'],
            'DIFF_EXAMPLE': DIFF_C,
        },
        'typescript': {
            'LANG_EXT': ['ts', 'tsx', 'js', 'jsx', 'xml', 'gradle'],
            'DIFF_EXAMPLE': DIFF_TYPESCRIPT,
        },
        'javascript': {
            'LANG_EXT': ['js', 'jsx', 'ts', 'tsx', 'xml', 'gradle'],
            'DIFF_EXAMPLE': DIFF_JAVASCRIPT,
        },
        
        # Multi-language configs for mobile dev
        # These handle repos where issues may touch multiple languages
        
        # Android (Java + Kotlin + build/resource files)
        'android': {
            'LANG_EXT': ['java', 'kt', 'kts', 'xml', 'gradle'],
            'DIFF_EXAMPLE': DIFF_KOTLIN
        },
        
        # React Native (TypeScript + JavaScript + native)
        'react_native': {
            'LANG_EXT': ['ts', 'tsx', 'js', 'jsx', 'java', 'kt', 'xml', 'gradle'],
            'DIFF_EXAMPLE': DIFF_TYPESCRIPT
        },
        
        # Flutter (Dart + native platform code)
        'flutter': {
            'LANG_EXT': ['dart', 'java', 'kt', 'xml', 'gradle', 'yaml'],
            'DIFF_EXAMPLE': DIFF_DART
        },
    }
    if language not in configs:
        raise RuntimeError(f'Unknown language {language}')
    return configs[language]


LANGUAGE = os.environ.get('SWEBENCH_LANG', 'python').lower()
STRUCTURE_KEYS = {'functions', 'classes', 'text'}
globals().update(get_config(LANGUAGE))
