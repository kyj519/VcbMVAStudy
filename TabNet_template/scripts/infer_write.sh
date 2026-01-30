#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "$ROOT_DIR/MVA.py" --working_mode 'infer' --input_model "$1" --input_root_file "$2" --branch_name "$3" --era "$4"
