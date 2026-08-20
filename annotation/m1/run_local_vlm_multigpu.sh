#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 MODEL_PATH MODEL_ID PROCESSOR_FAMILY MANIFEST RUN_DIR" >&2
  exit 2
fi

MODEL_PATH="$1"
MODEL_ID="$2"
PROCESSOR_FAMILY="$3"
MANIFEST="$4"
RUN_DIR="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${VLM_PYTHON:-python}"
PROMPT="${VLM_PROMPT:-${SCRIPT_DIR}/specification/prompt_v1.md}"
SCHEMA="${VLM_SCHEMA:-${SCRIPT_DIR}/specification/schemas/annotation_payload.schema.json}"
CANDIDATE_SCHEMA="${VLM_CANDIDATE_SCHEMA:-${SCRIPT_DIR}/specification/schemas/candidate_record.schema.json}"
MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-8192}"
CONSTRAINED_DECODING="${VLM_CONSTRAINED_DECODING:-none}"
RETRY_FAILED="${RETRY_FAILED:-0}"
read -r -a GPU_IDS <<< "${VLM_GPU_IDS:-3 4 5 7}"
NUM_SHARDS="${#GPU_IDS[@]}"

if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "VLM_GPU_IDS must contain at least one GPU ID" >&2
  exit 2
fi
if [[ "${PROCESSOR_FAMILY}" != "qwen35" && "${PROCESSOR_FAMILY}" != "internvl35" ]]; then
  echo "PROCESSOR_FAMILY must be qwen35 or internvl35" >&2
  exit 2
fi
if [[ "${CONSTRAINED_DECODING}" != "none" && "${CONSTRAINED_DECODING}" != "lmfe-json-schema" && "${CONSTRAINED_DECODING}" != "xgrammar-json-schema" ]]; then
  echo "VLM_CONSTRAINED_DECODING must be none, lmfe-json-schema, or xgrammar-json-schema" >&2
  exit 2
fi

retry_args=()
if [[ "${RETRY_FAILED}" == "1" ]]; then
  retry_args+=(--retry-failed)
fi

mkdir -p "${RUN_DIR}/logs"
pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[${shard_index}]}"
  log_path="${RUN_DIR}/logs/shard_${shard_index}.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${SCRIPT_DIR}${VLM_EXTRA_DEPS:+:${VLM_EXTRA_DEPS}}${PYTHONPATH:+:${PYTHONPATH}}" \
  HF_HOME="${PROJECT_ROOT}/.cache/huggingface" \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_local_vlm_manifest.py" \
    --project-root "${PROJECT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --processor-family "${PROCESSOR_FAMILY}" \
    --constrained-decoding "${CONSTRAINED_DECODING}" \
    --manifest "${MANIFEST}" \
    --prompt "${PROMPT}" \
    --schema "${SCHEMA}" \
    --candidate-schema "${CANDIDATE_SCHEMA}" \
    --output-dir "${RUN_DIR}" \
    --shard-index "${shard_index}" \
    --num-shards "${NUM_SHARDS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${retry_args[@]}" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "started shard=${shard_index} gpu=${gpu_id} pid=${pids[-1]} log=${log_path}"
done

shard_failure=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    shard_failure=1
  fi
done

PYTHONNOUSERSITE=1 \
PYTHONPATH="${SCRIPT_DIR}${VLM_EXTRA_DEPS:+:${VLM_EXTRA_DEPS}}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_m1_local_run.py" \
  --manifest "${MANIFEST}" \
  --schema "${SCHEMA}" \
  --candidate-schema "${CANDIDATE_SCHEMA}" \
  --run-dir "${RUN_DIR}"
summary_status="$?"

if [[ "${shard_failure}" -ne 0 || "${summary_status}" -ne 0 ]]; then
  exit 1
fi
