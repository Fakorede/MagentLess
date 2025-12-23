set -x

# api_key.sh
# export OPENAI_API_KEY=
# export OPENAI_BASE_URL=
# export OPENAI_MODEL=
# export OPENAI_EMBED_URL=
source script/api_key.sh

export PYTHONPATH=`pwd`
export TARGET_ID=
export NJ=1
export NUM_SETS=5
export NUM_SAMPLES_PER_SET=4
export NUM_REPRODUCTION=0
export FOLDER_NAME=mobiledev_bench_kotlin-claude-sonnet-4.5
export PROJECT_FILE_LOC=structure
export DATASET=local_json
export SPLIT=test

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================
# Single languages: java, kotlin, typescript, dart
# Multi-language configs: android, react_native, flutter
# ============================================================================NUM_SAMPLES_PER_SET
export SWEBENCH_LANG=kotlin

# ============================================================================
# LOCALIZATION PIPELINE
# ============================================================================
# Step 1.1: File-level localization (find relevant files)
./script/localization1.1.sh

# Step 1.2: Class/function-level localization
./script/localization1.2.sh

# Steps 1.3 and 1.4 - Embedding-based retrieval (now supports all languages)
./script/localization1.3.sh
./script/localization1.4.sh

# Step 2.1: Irrelevant file filtering
./script/localization2.1.sh

# Step 3.1-3.2: Fine-grained localization (using file_level_irrelevant output)
./script/localization3.1.sh
./script/localization3.2.sh

# ============================================================================
# REPAIR PIPELINE
# ============================================================================
./script/repair.sh

#./script/selection1.1.sh
#./script/selection1.2.sh
#./script/selection1.3.sh
#./script/selection2.1.sh
#./script/selection2.2.sh
#./script/selection2.3.sh
#./script/selection2.4.sh
./script/selection3.1.sh

#./script/evaluation.sh
