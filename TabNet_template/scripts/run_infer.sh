#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LIST_FILE="${ROOT_DIR}/infer_arg_list.txt"
LOG_DIR="${ROOT_DIR}/logs/logs_infer"
INFER_WRITE="${SCRIPT_DIR}/infer_write.sh"

mkdir -p "$LOG_DIR"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "[ERROR] $LIST_FILE not found."
  exit 1
fi

i=0
while IFS= read -r line || [[ -n "$line" ]]; do
  # 공백만 있는 줄 제거
  [[ -z "${line// }" ]] && continue
  # 주석 줄
  [[ "$line" =~ ^# ]] && continue

  i=$((i + 1))
  printf -v idx "%03d" "$i"
  log_file="${LOG_DIR}/infer_${idx}.log"

  echo "[INFO] [$idx] running: ${INFER_WRITE} $line"
  echo "       log -> ${log_file}"

  {
    echo "=== START $(date) ==="
    echo "CMD: ${INFER_WRITE} $line"
    echo

    # 여기서 에러 나도 다음 라인으로 넘어가게
    if ${INFER_WRITE} $line; then
      echo
      echo "[INFO] infer_write.sh finished successfully."
    else
      rc=$?
      echo
      echo "[ERROR] infer_write.sh failed with exit code ${rc}."
    fi

    echo
    echo "=== END $(date) ==="
  } >"$log_file" 2>&1

done < "$LIST_FILE"

echo
echo "[INFO] Done. Total ${i} runs. Logs in: ${LOG_DIR}/"
