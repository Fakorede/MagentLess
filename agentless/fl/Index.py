import copy
import os
from abc import ABC

import tiktoken
from llama_index.core import (
    Document,
    MockEmbedding,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.index_skeleton import parse_global_stmt_from_code
from agentless.util.preprocess_data import (
    clean_method_left_space,
    get_full_file_paths_and_classes_and_functions,
)
# MOBILE_DEV:
from get_repo_structure.get_repo_structure import (
    parse_python_file,
    parse_java_file,
    parse_go_file,
    parse_rust_file,
    parse_cpp_file,
    parse_typescript_file,
    parse_kotlin_file,
    parse_dart_file,
)


def parse_file_by_extension(file_name: str, content: str):
    """
    Parse a file using the appropriate parser based on file extension.
    Returns (class_info, function_names, file_lines).
    """
    file_lower = file_name.lower()
    
    try:
        if file_lower.endswith('.py'):
            return parse_python_file(None, content)
        elif file_lower.endswith('.java'):
            class_info, file_lines = parse_java_file(None, content)
            return class_info, [], file_lines
        elif file_lower.endswith('.go'):
            return parse_go_file(None, content)
        elif file_lower.endswith('.rs'):
            return parse_rust_file(None, content)
        elif file_lower.endswith(('.kt', '.kts')):
            return parse_kotlin_file(None, content)
        elif file_lower.endswith('.dart'):
            return parse_dart_file(None, content)
        elif file_lower.endswith(('.ts', '.tsx', '.js', '.jsx')):
            return parse_typescript_file(None, content)
        elif file_lower.endswith(('.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx')):
            return parse_cpp_file(None, content)
        else:
            # For unsupported files (xml, gradle, yaml, etc.), return empty structure
            return [], [], content.split('\n')
    except Exception:
        # If parsing fails, return empty structure
        return [], [], content.split('\n')


def construct_file_meta_data(file_name: str, clazzes: list, functions: list) -> dict:
    meta_data = {
        "file_name": file_name,
    }
    meta_data["File Name"] = file_name

    # MOBILE_DEV: Check if this is a TypeScript/JavaScript file
    is_typescript = file_name.lower().endswith(('.ts', '.tsx', '.js', '.jsx'))

    # MOBILE_DEV: Limit classes to avoid overly long metadata (TypeScript only)
    MAX_CLASSES = 30
    if clazzes:
        class_names = [c["name"] for c in clazzes]
        if is_typescript and len(class_names) > MAX_CLASSES:
            meta_data["Classes"] = ", ".join(class_names[:MAX_CLASSES]) + f" ... and {len(class_names) - MAX_CLASSES} more"
        else:
            meta_data["Classes"] = ", ".join(class_names)

    # MOBILE_DEV: Limit functions to avoid overly long metadata (TypeScript only)
    # Prioritize named functions over arrow_function_N
    MAX_FUNCTIONS = 30
    if functions:
        func_names = [f["name"] for f in functions]

        if is_typescript:
            # Separate named functions from arrow functions
            named_funcs = [name for name in func_names if not name.startswith("arrow_function_")]
            arrow_funcs = [name for name in func_names if name.startswith("arrow_function_")]

            # Include named functions first, then arrow functions
            if len(named_funcs) >= MAX_FUNCTIONS:
                # Too many named functions, only include first MAX_FUNCTIONS
                selected_funcs = named_funcs[:MAX_FUNCTIONS]
                meta_data["Functions"] = ", ".join(selected_funcs) + f" ... and {len(func_names) - len(selected_funcs)} more"
            else:
                # Include all named functions + some arrow functions
                remaining_slots = MAX_FUNCTIONS - len(named_funcs)
                selected_funcs = named_funcs + arrow_funcs[:remaining_slots]
                if len(selected_funcs) < len(func_names):
                    meta_data["Functions"] = ", ".join(selected_funcs) + f" ... and {len(func_names) - len(selected_funcs)} more"
                else:
                    meta_data["Functions"] = ", ".join(selected_funcs)
        else:
            # For non-TypeScript files, include all functions
            meta_data["Functions"] = ", ".join(func_names)

    return meta_data


def check_meta_data(meta_data: dict) -> bool:

    doc = Document(
        text="",
        metadata=meta_data,
        metadata_template="### {key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
    )

    if (
        num_tokens_from_messages(
            doc.get_content(metadata_mode=MetadataMode.EMBED),
            model="text-embedding-3-large",
        )
        > Settings.chunk_size // 2
    ):
        # half of the chunk size should not be metadata
        return False

    return True


def build_file_documents_simple(
    clazzes: list, functions: list, file_name: str, file_content: str
) -> list[Document]:
    """
    Really simple file document format, where we put all content of a single file into a single document
    """
    documents = []

    meta_data = construct_file_meta_data(file_name, clazzes, functions)

    doc = Document(
        text=file_content,
        metadata=meta_data,
        metadata_template="### {key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
    )
    doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
    doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
    if not check_meta_data(meta_data):
        # meta_data a bit too long, instead we just exclude meta data
        doc.excluded_embed_metadata_keys = list(meta_data.keys())
        doc.excluded_llm_metadata_keys = list(meta_data.keys())
        documents.append(doc)
    else:
        documents.append(doc)

    return documents


def build_file_documents_complex(
    clazzes: list, functions: list, file_name: str, file_content: str
) -> list[Document]:

    documents = []

    global_stmt, _ = parse_global_stmt_from_code(file_content)
    base_meta_data = construct_file_meta_data(file_name, clazzes, functions)

    for clazz in clazzes:
        content = "\n".join(clazz["text"])
        meta_data = copy.deepcopy(base_meta_data)
        meta_data["Class Name"] = clazz["name"]
        doc = Document(
            text=content,
            metadata=meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )

        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(meta_data):
            doc.excluded_embed_metadata_keys = list(meta_data.keys())
            doc.excluded_llm_metadata_keys = list(meta_data.keys())
        documents.append(doc)

        for class_method in clazz["methods"]:
            method_meta_data = copy.deepcopy(base_meta_data)
            method_meta_data["Class Name"] = clazz["name"]
            method_meta_data["Method Name"] = class_method["name"]
            content = clean_method_left_space("\n".join(class_method["text"]))

            doc = Document(
                text=content,
                metadata=method_meta_data,
                metadata_template="### {key}: {value}",
                text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
            )
            doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
            doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
            if not check_meta_data(method_meta_data):
                doc.excluded_embed_metadata_keys = list(method_meta_data.keys())
                doc.excluded_llm_metadata_keys = list(method_meta_data.keys())
            documents.append(doc)

    for function in functions:
        content = "\n".join(function["text"])
        function_meta_data = copy.deepcopy(base_meta_data)
        function_meta_data["Function Name"] = function["name"]
        doc = Document(
            text=content,
            metadata=function_meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )

        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(function_meta_data):
            doc.excluded_embed_metadata_keys = list(function_meta_data.keys())
            doc.excluded_llm_metadata_keys = list(function_meta_data.keys())
        documents.append(doc)

    if global_stmt != "":
        content = global_stmt
        global_meta_data = copy.deepcopy(base_meta_data)

        doc = Document(
            text=content,
            metadata=global_meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )
        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(global_meta_data):
            doc.excluded_embed_metadata_keys = list(global_meta_data.keys())
            doc.excluded_llm_metadata_keys = list(global_meta_data.keys())
        documents.append(doc)

    return documents


class EmbeddingIndex(ABC):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        persist_dir,
        filter_type,
        index_type,
        chunk_size,
        chunk_overlap,
        logger,
        **kwargs,
    ):
        self.instance_id = instance_id
        self.structure = structure
        self.problem_statement = problem_statement
        self.persist_dir = persist_dir + "/{instance_id}"
        self.filter_type = filter_type
        self.index_type = index_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        self.kwargs = kwargs
        # set some embedding global settings.

        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

    def filter_files(self, files):
        if self.filter_type == "given_files":
            given_files = self.kwargs["given_files"][: self.kwargs["filter_top_n"]]
            return given_files
        elif self.filter_type == "none":
            # all files are included
            return [file_content[0] for file_content in files]
        else:
            raise NotImplementedError

    def retrieve(self, mock=False):

        persist_dir = self.persist_dir.format(instance_id=self.instance_id)
        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("text-embedding-3-large").encode
        )
        if not os.path.exists(persist_dir) or mock:
            files, _, _ = get_full_file_paths_and_classes_and_functions(self.structure)
            filtered_files = self.filter_files(files)
            self.logger.info(f"Total number of considered files: {len(filtered_files)}")
            print(f"Total number of considered files: {len(filtered_files)}")
            documents = []

            for file_content in files:
                content = "\n".join(file_content[1])
                file_name = file_content[0]

                if file_name not in filtered_files:
                    continue

                # MOBILE_DEV:create documents using language-appropriate parser
                class_info, function_names, _ = parse_file_by_extension(file_name, content)
                if self.index_type == "simple":
                    docs = build_file_documents_simple(
                        class_info, function_names, file_name, content
                    )
                elif self.index_type == "complex":
                    docs = build_file_documents_complex(
                        class_info, function_names, file_name, content
                    )
                else:
                    raise NotImplementedError

                documents.extend(docs)

            self.logger.info(f"Total number of documents: {len(documents)}")
            print(f"Total number of documents: {len(documents)}")

            if mock:
                embed_model = MockEmbedding(
                    embed_dim=1024
                )  # embedding dimension does not matter for mocking.
                Settings.callback_manager = CallbackManager([token_counter])
            else:
                api_base = os.environ.get('OPENAI_EMBED_URL')
                # MOBILE_DEV:
                embed_model_name = os.environ.get('OPENAI_EMBED_MODEL', 'openai/text-embedding-3-large')
                embed_api_key = os.environ.get('OPENAI_EMBED_API_KEY')
                embed_model = OpenAIEmbedding(model_name=embed_model_name, api_base=api_base, api_key=embed_api_key)
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            #index.storage_context.persist(persist_dir=persist_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)

        self.logger.info(f"Retrieving with query:\n{self.problem_statement}")

        # MOBILE_DEV:Truncate problem statement to fit within embedding model's context length
        # text-embedding-3-large has max 8192 tokens, leave some margin
        max_tokens = 6000
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        tokens = encoding.encode(self.problem_statement)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
            self.logger.info(f"Truncated problem statement from {len(tokens)} to {max_tokens} tokens")
            query_text = truncated_text
        else:
            query_text = self.problem_statement

        retriever = VectorIndexRetriever(index=index, similarity_top_k=100)
        documents = retriever.retrieve(query_text)

        self.logger.info(
            f"Embedding Tokens: {token_counter.total_embedding_token_count}"
        )
        print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")

        traj = {
            "usage": {"embedding_tokens": token_counter.total_embedding_token_count}
        }

        token_counter.reset_counts()

        if mock:
            self.logger.info("Skipping since mock=True")
            return [], None, traj

        file_names = []
        meta_infos = []

        for node in documents:
            file_name = node.node.metadata["File Name"]
            if file_name not in file_names:
                file_names.append(file_name)
                self.logger.info("================")
                self.logger.info(file_name)

            self.logger.info(node.node.text)

            meta_infos.append({"code": node.node.text, "metadata": node.node.metadata})

        return file_names, meta_infos, traj
