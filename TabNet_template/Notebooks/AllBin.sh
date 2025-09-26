#!/usr/bin/env bash
set -euo pipefail


# 공통 옵션
U_CLIP_LOW=0.0
TCLIP_LOW=50
TCLIP_HIGH=100
METHOD=t_equal
NBINS=16
FINE=1024
MIN_NEFF_B=10
MIN_NEFF_S=10
BRANCH=template_score_MultiClass
MODE=b

# 입출력 루트
BASE="/gv0/Users/isyoon/Vcb_Post_Analysis/Sample"
OUTDIR="./XGBoost_CTag"
PLOTROOT="${OUTDIR}"
LOGDIR="${OUTDIR}/logs"
mkdir -p "$OUTDIR" "$LOGDIR"


# 가중치 JSON 자동 탐색 (환경변수 JSON이 비어있으면 최신 파일로)
JSON="${JSON:-}"
if [[ -z "$JSON" ]]; then
  JSON=$(ls -t ${OUTDIR}/summary_*.json | head -n1)
fi
if [[ -z "$JSON" || ! -f "$JSON" ]]; then
  echo "[ERR] weights JSON을 찾지 못했습니다. env JSON를 설정하거나 ./SPANeasfsat/summary_*.json가 존재하는지 확인하세요."
  exit 1
fi
echo "[INFO] Using weights JSON: $JSON"


run_one() {
  local YEAR="$1" CH="$2"

  local TTLJ="${BASE}/${YEAR}/Vcb_CTag/Central_Syst/Vcb_TTLJ_powheg.root:${CH}/Central/Result_Tree"
  local VCB="${BASE}/${YEAR}/Vcb_CTag/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root:${CH}/Central/Result_Tree"

  local TAG="${YEAR}_${CH}"
  local PLOTDIR="${PLOTROOT}/bins_${TAG}"
  local SAVEBINS="${OUTDIR}/bins_${TAG}.json"
  local LOG="${LOGDIR}/bins_${TAG}.log"

  mkdir -p "$PLOTDIR"

  echo "[RUN] ${TAG}"
  python BinningOptimizer.py \
    --u-clip-low "${U_CLIP_LOW}" \
    --t-clip-low "${TCLIP_LOW}" --t-clip-high "${TCLIP_HIGH}" \
    --mode "${MODE}" \
    --ttlj "${TTLJ}" \
    --vcb  "${VCB}" \
    --weights-json "${JSON}" \
    --branch-name "${BRANCH}" \
    --method "${METHOD}" --nbins "${NBINS}" --fine "${FINE}" \
    --min-neff-b "${MIN_NEFF_B}" --min-neff-s "${MIN_NEFF_S}" \
    --plot-dir "${PLOTDIR}" \
    --save-bins "${SAVEBINS}" \
    |& tee "${LOG}"

  echo "[DONE] ${TAG} -> ${SAVEBINS}"
}

YEARS=(2016preVFP 2016postVFP 2017 2018)
CHS=(Mu El)

for Y in "${YEARS[@]}"; do
  for C in "${CHS[@]}"; do
    run_one "$Y" "$C"
  done
done

echo "[ALL DONE] outputs under ${OUTDIR}"
