#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

cd "$PROJECT_DIR"
source setup.sh
micromamba activate pytorch
python "$ROOT_DIR/MVA.py" --working_mode plot --input_model "$1" --checkpoint "$2"
