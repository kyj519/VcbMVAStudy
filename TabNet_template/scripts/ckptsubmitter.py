import os, re, htcondor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

base_path = "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/TabNET_model/largePhaseSpace_B_MultiClass"
ckpt_list = sorted(
    os.path.join(base_path + "/checkpoints", f)
    for f in os.listdir(base_path + "/checkpoints")
    if f.startswith("model_epoch") and f.endswith(".zip")
)
ckpt_list = ckpt_list[200:270]
log_path = os.path.join(base_path, "log")
os.makedirs(log_path, exist_ok=True)

def parse_epoch(path):
    m = re.search(r"model_epoch(\d+)\.zip$", os.path.basename(path))
    return int(m.group(1)) if m else -1
ncpu = 4
submit_desc = {
    "executable": str(SCRIPT_DIR / "ckptplotter.sh"),
    "arguments": '"$(BASE_PATH) $(CKPT)"',                      # 체크포인트 파일 전체 경로가 하나의 인자
    "output":   f"{log_path}/ckptplotter_$(EPOCH).out",
    "error":    f"{log_path}/ckptplotter_$(EPOCH).err",
    "log":      f"{log_path}/ckptplotter_$(EPOCH).log",
    "request_cpus": ncpu,
    "request_memory": "16GB",
    "universe": "vanilla",
    "getenv":   "True",
    "environment": (
        f"NTHREADS={ncpu} "
        f"OMP_NUM_THREADS={ncpu} "
        f"OPENBLAS_NUM_THREADS={ncpu} "
        f"MKL_NUM_THREADS={ncpu} "
        f"BLIS_NUM_THREADS={ncpu} "
        f"GOTO_NUM_THREADS={ncpu} "
        f"NUMEXPR_NUM_THREADS={ncpu} "
        f"VECLIB_MAXIMUM_THREADS={ncpu} "
        f"MKL_DYNAMIC=FALSE "
        f"OMP_PROC_BIND=true "
        f"OMP_PLACES=cores "
        f"OMP_THREAD_LIMIT={ncpu}"
    ),
    # 필요시 자원/전송정책 지정:
    # "request_cpus": "1",
    # "request_memory": "2 GB",
    # "should_transfer_files": "YES",
    # "when_to_transfer_output": "ON_EXIT",
}

schedd = htcondor.Schedd()
sub = htcondor.Submit(submit_desc)

# itemdata로 (체크포인트 경로, 에폭) 바인딩
items = ({"BASE_PATH": base_path, "CKPT": ck, "EPOCH": f"{parse_epoch(ck):03d}"} for ck in ckpt_list)

with schedd.transaction() as txn:
    cluster_id = sub.queue_with_itemdata(txn, 1, items)
    print(f"Submitted {len(ckpt_list)} jobs in cluster {cluster_id}")
