#!/bin/bash
set -e

REPO_DIR="repo"


# repositories that need re-cloning
REPOS=(
    "artsy/eigen:eigen"
    "pagopa/io-app:io-app"
    "NMF-earth/nmf-app:nmf-app"
    "mattermost/mattermost-mobile:mattermost-mobile"
)

echo "Re-cloning repositories with PR refs..."

for repo_entry in "${REPOS[@]}"; do
    IFS=':' read -r repo_url local_name <<< "$repo_entry"

    echo ""
    echo "=========================================="
    echo "Processing: $repo_url -> $local_name"
    echo "=========================================="

    # Remove existing directory
    if [ -d "$REPO_DIR/$local_name" ]; then
        echo "Removing existing $REPO_DIR/$local_name..."
        rm -rf "$REPO_DIR/$local_name"
    fi

    # Clone the repository
    echo "Cloning https://github.com/$repo_url.git..."
    git clone "https://github.com/$repo_url.git" "$REPO_DIR/$local_name"

    # Fetch all PR refs
    echo "Fetching PR refs (this may take a while)..."
    cd "$REPO_DIR/$local_name"
    git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'
    cd ../..

    echo "Done with $local_name"
done

echo ""
echo "All repositories re-cloned successfully!"
echo ""
echo "You can now run: python3 script/generate_structure.py"
