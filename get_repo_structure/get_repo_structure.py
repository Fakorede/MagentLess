import argparse
import ast
import json
import os
import subprocess
import uuid
from typing import List, Generator, Union

import pandas as pd
from tqdm import tqdm
from tree_sitter import Node
from tree_sitter_language_pack import get_parser

repo_to_top_folder = {
    # Java
    "antennapod/antennapod": "antennapod",

    # Kotlin
    "element-hq/element-x-android": "element-x-android",
    "jackeblan/geto": "geto",
    "lemmynet/jerboa": "jerboa",
    "mjaakko/neostumbler": "neostumbler",
    "openhab/openhab-android": "openhab-android",
    "streetcomplete/streetcomplete": "streetcomplete",
    "thunderbird/thunderbird-android": "thunderbird-android",
    "tuskyapp/tusky": "tusky",
    "paulwoitaschek/voice": "voice",
    "commons-app/apps-android-commons": "apps-android-commons",
    "wordpress-mobile/wordpress-android": "wordpress-android",
    "futsch1/medtimer": "medtimer",
    
    # Dart/Flutter
    "palisadoesfoundation/talawa": "talawa",
    "zulip/zulip-flutter": "zulip-flutter",

    # TypeScript
    "expensify/app": "expensify",
    "metamask/metamask-mobile": "metamask-mobile",
    "rocketchat/rocket.chat.reactnative": "rocket.chat.reactnative",
}


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clone_repo(repo_name, repo_playground):
    try:
        # Normalize repo name to lowercase for lookup
        repo_key = repo_name.lower()
        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{repo_to_top_folder[repo_key]}..."
        )
        dir_name = repo_to_top_folder[repo_key]
        subprocess.run(
            [
                'cp',
                f'repo/{dir_name}',
                f'{repo_playground}/{dir_name}',
                '-r',
            ],
            check=True,
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    # Normalize repo name to lowercase for lookup
    repo_key = repo_name.lower()
    clone_repo(repo_name, repo_playground)
    checkout_commit(f"{repo_playground}/{repo_to_top_folder[repo_key]}", commit_id)
    structure = create_structure(f"{repo_playground}/{repo_to_top_folder[repo_key]}")
    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{repo_to_top_folder[repo_key]}"], check=True
    )
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def traverse(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def get_child(node: Node, type_name: str, skip: int = 0) -> Union[Node, None]:
    for child in node.children:
        if child.type == type_name:
            if skip == 0:
                return child
            skip = skip - 1
    return None


def get_child_chain(node: Node, type_names: List[str]) -> Union[str, None]:
    for type_name in type_names:
        node = get_child(node, type_name)
        if node is None:
            return node
    return node


def get_name(node: Node, type_name: str = 'identifier') -> Union[str, None]:
    child = get_child(node, type_name)
    if child is None:
        return None
    return child.text.decode('utf-8')


def parse_java_file(file_path, file_content=None):
    """Parse a Java file to extract interface definitions and class definitions with their line numbers.
    :param file_path: Path to the Java file.
    :return: Class names, and file contents
    """
    parser = get_parser('java')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], ""

    class_info = []

    for node in traverse(tree.root_node):
        if node.type == "interface_declaration" or node.type == "class_declaration":
            info = None
            if node.type == "interface_declaration":
                info = class_info
            elif node.type == "class_declaration":
                info = class_info

            node_name = get_name(node)
            if node_name is None:
                continue

            methods = []
            for n in traverse(node):
                if n.type == "method_declaration":
                    method_name = get_name(n)
                    if method_name is None:
                        continue
                    methods.append(
                        {
                            "name": method_name,
                            "start_line": n.start_point.row,
                            "end_line": n.end_point.row,
                            "text": n.text.decode('utf-8').splitlines(),
                        }
                    )
            info.append(
                {
                    "name": node_name,
                    "start_line": node.start_point.row,
                    "end_line": node.end_point.row,
                    "text": node.text.decode('utf-8').splitlines(),
                    "methods": methods,
                }
            )

    return class_info, file_content.splitlines()


def parse_go_file(file_path, file_content=None):
    """Parse a Go file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Go file.
    :return: Class names, function names, and file contents
    """
    parser = get_parser('go')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []

    for node in traverse(tree.root_node):
        if node.type == "type_declaration":
            type_spec = get_child(node, 'type_spec')
            if type_spec is None:
                continue
            name = get_name(type_spec, 'type_identifier')
            if name is None:
                continue
            methods = []
            class_info.append({
                'name': name,
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
                'methods': methods,
            })
        elif node.type == 'method_declaration':
            method_name = get_name(node, 'field_identifier')
            if method_name is None:
                continue
            function_names.append({
                'name': method_name,
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
            })
        elif node.type == 'function_declaration':
            func_name = get_name(node, 'identifier')
            if func_name is None:
                continue
            function_names.append({
                'name': func_name,
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
            })

    return class_info, function_names, file_content.splitlines()


def parse_rust_file(file_path, file_content=None):
    """Parse a Rust file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Rust file.
    :return: Class names, function names, and file contents
    """
    parser = get_parser('rust')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_to_methods = {}

    def get_type(node: Node):
        if node is None:
            return None
        if node.type == 'type_identifier':
            return node.text.decode('utf-8')
        elif node.type == 'generic_type':
            type_node = node.child_by_field_name('type')
            if type_node is None:
                return None
            return get_type(type_node)
        return None

    for node in traverse(tree.root_node):
        if node.type == 'struct_item' or node.type == 'enum_item':
            name = get_name(node, 'type_identifier')
            if name is None:
                continue
            methods = []
            class_to_methods[name] = methods
            class_info.append({
                'name': name,
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
                'methods': methods,
            })
        elif node.type == 'impl_item':
            class_ = get_type(node.child_by_field_name('type'))
            methods = class_to_methods.get(class_, None)
            if methods is not None:
                for child in traverse(node):
                    if child.type == 'function_item':
                        func_name = get_name(child)
                        if func_name is None:
                            continue
                        methods.append({
                            'name': func_name,
                            'start_line': child.start_point.row,
                            'end_line': child.end_point.row,
                            'text': child.text.decode('utf-8').splitlines(),
                        })
        elif node.type == 'function_item':
            func_name = get_name(node)
            if func_name is None:
                continue
            function_names.append({
                'name': func_name,
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
            })

    return class_info, function_names, file_content.splitlines()


def parse_cpp_file(file_path, file_content=None):
    """Parse a C/C++ file to extract class and function definitions with their line numbers.
    :param file_path: Path to the C/C++ file.
    :return: Class names, function names, and file contents
    """
    parser = get_parser('cpp')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []

    def get_type(node: Node):
        if node is None:
            return None
        if node.type == 'type_identifier':
            return node.text.decode('utf-8')
        elif node.type == 'template_type':
            name_node = node.child_by_field_name('name')
            if name_node is None:
                return None
            return get_type(name_node)
        return None

    for node in traverse(tree.root_node):
        if node.type == 'class_specifier':
            methods = []
            if file_path.endswith('.c'):
                continue
            class_name = get_type(node.child_by_field_name('name'))
            if class_name is None:
                continue
            class_info.append({
                'name': class_name,
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
                'methods': methods,
            })
            for child in traverse(node):
                if child.type == 'function_definition':
                    name_node = child.child_by_field_name('declarator')
                    if name_node is None:
                        continue
                    name_node = name_node.child_by_field_name('declarator')
                    if name_node is None:
                        continue
                    methods.append({
                        'name': name_node.text.decode('utf-8'),
                        'start_line': child.start_point.row,
                        'end_line': child.end_point.row,
                        'text': child.text.decode('utf-8').splitlines(),
                    })
        elif node.type == 'function_definition':
            name_node = node.child_by_field_name('declarator')
            if name_node is None:
                continue
            name_node = name_node.child_by_field_name('declarator')
            if name_node is None:
                continue

            in_class = False
            tmp = node
            while tmp != tree.root_node:
                if tmp.type == 'class_specifier':
                    in_class = True
                    break
                tmp = tmp.parent
            if in_class:
                continue

            function_names.append({
                'name': name_node.text.decode('utf-8'),
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
            })

    return class_info, function_names, file_content.splitlines()


def parse_typescript_file(file_path, file_content=None):
    """Parse a TypeScript file to extract interface definitions and class definitions with their line numbers.
    :param file_path: Path to the TypeScript file.
    :return: Class names, function names, and file contents
    """
    parser = get_parser('typescript')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    arrow_function_idx = 0

    for node in traverse(tree.root_node):
        if node.type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            if name_node is None:
                continue
            methods = []
            class_info.append({
                'name': name_node.text.decode('utf-8'),
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
                'methods': methods,
            })
            for child in traverse(node):
                if child.type == 'method_definition':
                    method_name = child.child_by_field_name('name')
                    if method_name is None:
                        continue
                    methods.append({
                        'name': method_name.text.decode('utf-8'),
                        'start_line': child.start_point.row,
                        'end_line': child.end_point.row,
                        'text': child.text.decode('utf-8').splitlines(),
                    })
        elif node.type == 'function_declaration':
            func_name = node.child_by_field_name('name')
            if func_name is None:
                continue
            function_names.append({
                'name': func_name.text.decode('utf-8'),
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
            })
        elif node.type == 'arrow_function':
            function_names.append({
                'name': f'arrow_function_{arrow_function_idx}',
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
            })
            arrow_function_idx = arrow_function_idx + 1

    return class_info, function_names, file_content.splitlines()


def parse_kotlin_file(file_path, file_content=None):
    """Parse a Kotlin file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Kotlin file.
    :return: Class names, function names, and file contents
    """
    parser = get_parser('kotlin')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []

    for node in traverse(tree.root_node):
        if node.type == 'class_declaration':
            name_node = None
            for child in node.children:
                if child.type == 'type_identifier':
                    name_node = child
                    break
            if name_node is None:
                continue
            
            methods = []
            for child in traverse(node):
                if child.type == 'function_declaration':
                    func_name_node = None
                    for c in child.children:
                        if c.type == 'simple_identifier':
                            func_name_node = c
                            break
                    if func_name_node:
                        methods.append({
                            'name': func_name_node.text.decode('utf-8'),
                            'start_line': child.start_point.row,
                            'end_line': child.end_point.row,
                            'text': child.text.decode('utf-8').splitlines(),
                        })
            
            class_info.append({
                'name': name_node.text.decode('utf-8'),
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
                'methods': methods,
            })
        elif node.type == 'function_declaration':
            # Check if this is a top-level function (not inside a class)
            in_class = False
            tmp = node.parent
            while tmp is not None:
                if tmp.type == 'class_declaration':
                    in_class = True
                    break
                tmp = tmp.parent
            
            if not in_class:
                func_name_node = None
                for child in node.children:
                    if child.type == 'simple_identifier':
                        func_name_node = child
                        break
                if func_name_node:
                    function_names.append({
                        'name': func_name_node.text.decode('utf-8'),
                        'start_line': node.start_point.row,
                        'end_line': node.end_point.row,
                        'text': node.text.decode('utf-8').splitlines(),
                    })

    return class_info, function_names, file_content.splitlines()


def parse_dart_file(file_path, file_content=None):
    """Parse a Dart file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Dart file.
    :return: Class names, function names, and file contents
    """
    parser = get_parser('dart')

    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                file_content = file.read()
                tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            tree = parser.parse(bytes(file_content, "utf-8"))
        except Exception as e:
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []

    for node in traverse(tree.root_node):
        if node.type == 'class_definition':
            name_node = node.child_by_field_name('name')
            if name_node is None:
                continue
            
            methods = []
            for child in traverse(node):
                if child.type == 'method_signature' or child.type == 'function_signature':
                    method_name = child.child_by_field_name('name')
                    if method_name:
                        methods.append({
                            'name': method_name.text.decode('utf-8'),
                            'start_line': child.start_point.row,
                            'end_line': child.end_point.row,
                            'text': child.text.decode('utf-8').splitlines(),
                        })
            
            class_info.append({
                'name': name_node.text.decode('utf-8'),
                'start_line': node.start_point.row,
                'end_line': node.end_point.row,
                'text': node.text.decode('utf-8').splitlines(),
                'methods': methods,
            })
        elif node.type == 'function_signature' or node.type == 'function_definition':
            # Check if this is a top-level function
            in_class = False
            tmp = node.parent
            while tmp is not None:
                if tmp.type == 'class_definition':
                    in_class = True
                    break
                tmp = tmp.parent
            
            if not in_class:
                func_name = node.child_by_field_name('name')
                if func_name:
                    function_names.append({
                        'name': func_name.text.decode('utf-8'),
                        'start_line': node.start_point.row,
                        'end_line': node.end_point.row,
                        'text': node.text.decode('utf-8').splitlines(),
                    })

    return class_info, function_names, file_content.splitlines()


def check_file_ext(file_name, language):
    exts = {
        'cpp': ['h', 'hpp', 'hxx', 'c', 'cpp', 'cc', 'cxx'],
        'typescript': ['js', 'ts'],
    }
    file_name = file_name.lower()
    for ext in exts[language]:
        if file_name.endswith(f'.{ext}'):
            return True
    return False


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            elif file_name.endswith('.java'):
                file_path = os.path.join(root, file_name)
                class_info, file_lines = parse_java_file(file_path)
                curr_struct[file_name] = {
                    'classes': class_info,
                    'functions': [],
                    'text': file_lines,
                }
            elif file_name.endswith('.go'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_go_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            elif file_name.endswith('.rs'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_rust_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            elif check_file_ext(file_name, 'cpp'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_cpp_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            elif check_file_ext(file_name, 'typescript'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_typescript_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            elif file_name.endswith('.kt') or file_name.endswith('.kts'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_kotlin_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            elif file_name.endswith('.dart'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_dart_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure

