#!/bin/bash
# export PASSPHRASE=passphrase
# chmod +x ./scripts/bash/00_cicd_pipeline.sh
# ./scripts/bash/00_cicd_pipeline.sh

# CREATING A NEW FEATURE BRANCH
# git checkout main
# git status -> working directory should be clean
# git checkout -b $NEW_BRANCH -> create new branch
# git push origin $NEW_BRANCH
# git fetch origin $NEW_BRANCH

# Set variables
FEATURE_BRANCH="feature/modeling"

# Add, commit and push final changes
git status
git add .
git commit -m "Final commit."
git pull origin $FEATURE_BRANCH
git push origin $FEATURE_BRANCH

# Run unit & integrity Tests
chmod +x ./scripts/bash/01_tests_running.sh
./scripts/bash/01_tests_running.sh

# Build & push docker images
chmod +x ./scripts/bash/02_image_building.sh
./scripts/bash/02_image_building.sh

# Sincronize buckets
.pytradex_venv/bin/python scripts/env_sincronizing/env_sincronizing.py --source_bucket pytradex-dev --destination_bucket pytradex-prod

# Checkout to main
git checkout main

# Pull origin main
git pull origin main

# Merge feature branch (solve merge conflicts if needed)
git merge $FEATURE_BRANCH

# Change env in config.yml before commiting changes (manually)

# Commit changes to main
git commit -m "Merge $FEATURE_BRANCH into main"

# Push changes to main
git push origin main

# Delete the local feature branch
git branch -d $FEATURE_BRANCH

# Delete remote branch on CodeCommit
git push origin --delete $FEATURE_BRANCH