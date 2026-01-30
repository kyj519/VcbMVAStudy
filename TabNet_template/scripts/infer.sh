#!/usr/bin/env bash
# batch_infer_with_logging_2016postVFP.sh
# 2016postVFP용: infer_write.sh 실행 + per-job 로그 + 요약 TSV + 실패 재실행 스크립트

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_WRITE="${SCRIPT_DIR}/infer_write.sh"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/infer_logs}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_2016postVFP"
OUT_DIR="$LOG_ROOT/$RUN_TAG"
mkdir -p "$OUT_DIR"

SUMMARY_TSV="$OUT_DIR/summary.tsv"
echo -e "ts_start\tts_end\telapsed_s\tstatus\tcode\tera\tmodel_variant\tsample\tmodel_dir\tout_tag\tcmd" > "$SUMMARY_TSV"

log_job() {
  # infer_write.sh <model_dir> <sample.root> <out_tag> <era>
  local model_dir="$1"
  local sample_path="$2"
  local out_tag="$3"
  local era="$4"

  local sample_base; sample_base="$(basename "$sample_path" .root)"
  local model_variant="unknown"
  [[ "$out_tag" == *6Class* ]] && model_variant="6Class"
  [[ "$out_tag" == *7Class* ]] && model_variant="7Class"

  local ts_start_iso ts_end_iso ts_start ts_end elapsed rc status job_id log_file
  ts_start_iso="$(date -Is)"
  ts_start="$(date +%s)"

  job_id="${era}_${model_variant}_${sample_base}"
  log_file="$OUT_DIR/${job_id}.log"

  if [[ ! -f "$sample_path" ]]; then
    echo "[WARN] Input not found: $sample_path" | tee -a "$log_file"
  fi

  echo "=== RUN $job_id ===" | tee -a "$log_file"
  echo "cmd: ${INFER_WRITE} \"$model_dir\" \"$sample_path\" \"$out_tag\" \"$era\"" | tee -a "$log_file"
  echo "started: $ts_start_iso" | tee -a "$log_file"

  set +e
  stdbuf -oL -eL ${INFER_WRITE} "$model_dir" "$sample_path" "$out_tag" "$era" 2>&1 | tee -a "$log_file"
  rc="${PIPESTATUS[0]}"
  set -e

  ts_end_iso="$(date -Is)"
  ts_end="$(date +%s)"
  elapsed=$(( ts_end - ts_start ))
  status="OK"; [[ "$rc" -ne 0 ]] && status="FAIL"

  printf "%s\t%s\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$ts_start_iso" "$ts_end_iso" "$elapsed" "$status" "$rc" "$era" "$model_variant" "$sample_base" "$model_dir" "$out_tag" \
    "${INFER_WRITE} \"$model_dir\" \"$sample_path\" \"$out_tag\" \"$era\"" >> "$SUMMARY_TSV"

  echo "finished: $ts_end_iso  => $status (rc=$rc)  elapsed=${elapsed}s" | tee -a "$log_file"
  echo >> "$log_file"
}

# 공통 모델 경로
MODEL_6="/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/TabNET_model/largePhaseSpace_B_MultiClass_EarlyStopLoss"
MODEL_7="/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/TabNET_model/largePhaseSpace_B_MultiClass_7Class/"
ERA="2016postVFP"
BASE="/gv0/Users/isyoon/Vcb_Post_Analysis/Sample/${ERA}/Vcb_BTag"

# =========================
# 2016postVFP (pre 리스트를 post로 전환)
# =========================
log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT2000toInf.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT2000toInf.root" "template_score_7Class" "$ERA"

log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT200to300.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT200to300.root" "template_score_7Class" "$ERA"

log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT300to500.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT300to500.root" "template_score_7Class" "$ERA"

# NOTE: HT500to700 (오타 't' 제거)
log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT500to700.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_QCD_bEnriched_HT500to700.root" "template_score_7Class" "$ERA"

log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_TTLL_powheg.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_TTLL_powheg.root" "template_score_7Class" "$ERA"

# NOTE: WJets 파일명 트레일링 '.' 없이
log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_WJets_HT1200to2500.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_WJets_HT1200to2500.root" "template_score_7Class" "$ERA"

log_job "$MODEL_6" "${BASE}/Central_Syst/Vcb_WJets_HT600to800.root" "template_score_6Class" "$ERA"
log_job "$MODEL_7" "${BASE}/Central_Syst/Vcb_WJets_HT600to800.root" "template_score_7Class" "$ERA"

# ---------- 요약 ----------
echo
echo "Summary file: $SUMMARY_TSV"
total=$(($(wc -l < "$SUMMARY_TSV") - 1))
ok=$(grep -c -P "\tOK\t" "$SUMMARY_TSV" || true)
fail=$(grep -c -P "\tFAIL\t" "$SUMMARY_TSV" || true)
echo "Total jobs: $total  OK: $ok  FAIL: $fail"
echo

# 실패 재실행 스크립트 생성
if [[ "$fail" -gt 0 ]]; then
  RERUN="$OUT_DIR/rerun_failures.sh"
  awk -F'\t' 'NR>1 && $4=="FAIL" {print $11}' "$SUMMARY_TSV" > "$OUT_DIR/.fail_cmds.txt"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo '# Re-run failed jobs captured earlier'
    while IFS= read -r line; do
      echo "$line"
    done < "$OUT_DIR/.fail_cmds.txt"
  } > "$RERUN"
  chmod +x "$RERUN"
  echo "Created rerun script for failures: $RERUN"
fi
