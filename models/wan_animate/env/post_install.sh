#!/usr/bin/env bash
# SAM2's own pip install ships an EMPTY sam2_configs package (its setup.py never
# declares the YAML files as package_data), so every replace-mode preprocessing
# call fails with "hydra.errors.MissingConfigException: Cannot find primary
# config 'sam2_hiera_l.yaml'". Restore the 4 YAML files from the sam2 repo at
# the pinned commit (0e78a11) into the installed package.
# See docs/models/wan_animate/README.md section 2.1.1.
# Run by scripts/build_env.sh with REPO_ROOT, DATA_DIR and ENV_DIR exported.
set -euo pipefail
DEST="$(echo "$ENV_DIR"/lib/python*/site-packages/sam2_configs)"
[ -d "$DEST" ] || { echo "sam2_configs package not found in $ENV_DIR" >&2; exit 1; }
cp -v "$REPO_ROOT"/models/wan_animate/env/sam2_configs/*.yaml "$DEST"/
