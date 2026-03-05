#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-zai-org/GLM-Image}"
INPUT_IMAGE="${1:-input.png}"
PROMPT="${PROMPT:-Turn this photo into an anime illustration while preserving composition.}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-output_glm_edit.png}"

python minimal_edit.py \
  --model-path "${MODEL_PATH}" \
  --image "${INPUT_IMAGE}" \
  --prompt "${PROMPT}" \
  --output "${OUTPUT_IMAGE}"
