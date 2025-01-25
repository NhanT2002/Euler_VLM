#!/bin/bash

# Set default commit message if not provided
COMMIT_MSG=${1:-"Auto-commit: $(date)"}

# Add, commit, and push
git add .
git commit -m "$COMMIT_MSG"
git push
