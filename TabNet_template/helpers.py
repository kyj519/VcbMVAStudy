from contextlib import contextmanager, nullcontext
import importlib
import re
import torch
import numpy as np
from typing import Iterable, List, Optional, Dict, Any, Tuple
from time import perf_counter
from pytorch_tabnet.utils import PredictDataset, SparsePredictDataset
from torch.utils.data import DataLoader, TensorDataset
import scipy
import scipy.sparse as sp
from pytorch_tabnet.callbacks import Callback
import os, sys
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# device-aware loader defaults
def _resolve_loader_params(
    device,
    num_workers: Optional[int],
    pin_memory: Optional[bool],
    prefetch_factor: Optional[int],
    persistent_workers: Optional[bool],
):
    use_cuda = (
        isinstance(device, torch.device)
        and device.type == "cuda"
        or (isinstance(device, str) and str(device).startswith("cuda"))
    )
    env_workers = os.getenv("DL_WORKERS")
    if num_workers is None:
        try:
            num_workers = max(0, int(env_workers)) if env_workers is not None else (4 if use_cuda else 0)
        except Exception:
            num_workers = 4 if use_cuda else 0
        try:
            import multiprocessing as mp
            if mp.current_process().daemon:
                num_workers = 0
        except Exception:
            pass
    num_workers = max(0, int(num_workers))

    if pin_memory is None:
        pin_memory = use_cuda
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if prefetch_factor is None:
        prefetch_factor = 4 if num_workers > 0 else None
    pf = prefetch_factor if num_workers > 0 else None

    kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory and use_cuda,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    if pf is not None and num_workers > 0:
        kwargs["prefetch_factor"] = pf
    return kwargs, use_cuda


def _resolve_batch_size(
    model,
    batch_size,
    env_key: str = "BATCH_INFER",
    gpu_default: int = 8192*2,
    cpu_default: int = 32,
):
    if batch_size is not None:
        try:
            return int(batch_size)
        except Exception:
            pass
    env_bs = os.getenv(env_key)
    if env_bs is not None:
        try:
            return int(env_bs)
        except Exception:
            pass
    m_bs = getattr(model, "batch_size", None)
    if m_bs is not None:
        try:
            return int(m_bs)
        except Exception:
            pass
    dev = getattr(model, "device", None)
    use_cuda = (
        isinstance(dev, torch.device)
        and dev.type == "cuda"
        or (isinstance(dev, str) and str(dev).startswith("cuda"))
    )
    return int(gpu_default if use_cuda else cpu_default)


def _load_training_config_module(config_path: Optional[str]):
    """
    config_path가 주어지면 해당 경로의 파이썬 파일(예: TrainingConfig.py)을 모듈로 로드.
    없으면 기존 'import TrainingConfig' 모듈을 반환.
    """
    if config_path is None:
        import TrainingConfig as _TC

        return _TC

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"TrainingConfig file not found: {config_path}")

    mod_name = "TrainingConfig_Dynamic"
    spec = importlib.util.spec_from_file_location(mod_name, config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec and spec.loader, "Invalid module spec for TrainingConfig"
    spec.loader.exec_module(mod)
    return mod


def _csr_to_torch_sparse_coo(csr: sp.csr_matrix, dtype=np.float32):
    coo = csr.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64)
    values = coo.data.astype(dtype, copy=False)
    i = torch.from_numpy(indices)
    v = torch.from_numpy(values)
    return torch.sparse_coo_tensor(
        i, v, size=coo.shape, dtype=torch.float32, device="cpu"
    )


def _ensure_reducing_t(model, device):
    if hasattr(model, "_reducing_t") and model._reducing_t is not None:
        Rt = model._reducing_t
        if Rt.device != device:
            model._reducing_t = Rt.to(device)
        return model._reducing_t

    R = getattr(model, "reducing_matrix", None)
    if R is None:
        raise ValueError("model.reducing_matrix가 필요합니다.")

    if sp.issparse(R):
        R_csr = R.tocsr()
        Rt = _csr_to_torch_sparse_coo(R_csr)
    else:
        R_csr = sp.csr_matrix(np.asarray(R, dtype=np.float32))
        Rt = _csr_to_torch_sparse_coo(R_csr)

    model._reducing_t = Rt.to(device)
    return model._reducing_t


@contextmanager
def task(name: str):
    t0 = perf_counter()
    print(f"▶ Starting {name} ...")
    try:
        yield
    finally:
        dt = perf_counter() - t0
        print(f"✓ Finished {name} in {dt:.2f} seconds\n")


def _iter_with_rich_progress(loader, desc: str):
    """
    Wrap an iterable with a rich progress bar that stacks cleanly
    when multiple bars are shown.
    """
    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    try:
        total = len(loader)
    except Exception:
        total = None

    with Progress(*columns, refresh_per_second=5, transient=True, expand=True) as progress:
        task_id = progress.add_task(desc, total=total)
        for item in loader:
            yield item
            progress.advance(task_id)


def pick_best_device(min_free_gb: float = 1.0, blacklist: tuple[int, ...] = ()):
    """
    가장 '남는 메모리'가 큰 GPU를 고르고, 없으면 CPU로.
    - min_free_gb: 이만큼 이상 여유 메모리 있는 GPU만 후보로
    - blacklist: 피하고 싶은 GPU 인덱스들 (예: (0,) )
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        if i in blacklist:
            continue
        # 대부분 버전에서 안전한 방식: 컨텍스트로 디바이스 전환 후 mem_get_info()
        with torch.cuda.device(i):
            free_b, total_b = torch.cuda.mem_get_info()
        if free_b > best_free and free_b >= min_free_gb * (1024**3):
            best_idx, best_free = i, free_b

    if best_idx is None:
        # 기준을 만족하는 GPU가 없으면, 그 중 가장 여유 있는 것 또는 CPU로
        # (아래는 그냥 CPU로 내리는 보수적 선택)
        return torch.device("cpu")
    return torch.device(f"cuda:{best_idx}")


def compute_class_counts(
    y: np.ndarray, w: Optional[np.ndarray] = None, num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Class-balanced focal loss용으로 클래스별 '가중합'을 계산.
    - y: 정수 라벨 (0..K-1)
    - w: 이벤트 가중치 (없으면 1)
    - num_classes: K (생략 시 y의 유니크 라벨로 추정)
    반환: shape=(K,) float64
    """
    y = np.asarray(y).astype(int)
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    if w is None:
        w = np.ones_like(y, dtype=np.float64)
    counts = np.bincount(y, weights=w, minlength=num_classes).astype(np.float64)
    # 0 분모 방지용 작은 epsilon
    counts[counts <= 0.0] = 1e-12
    return counts


def predict_proba_fast(
    model,
    X,
    batch_size=None,
    num_workers=None,
    amp=True,
    amp_dtype=torch.float16,
    pin_memory=None,
    prefetch_factor=None,
    persistent_workers=None,
    return_numpy=True,
):

    # --- match original behavior ---
    model.network.eval()

    # DataLoader construction (reuse the same datasets as original)
    if scipy.sparse.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    bs = _resolve_batch_size(model, batch_size)

    loader_kwargs, use_cuda = _resolve_loader_params(
        model.device, num_workers, pin_memory, prefetch_factor, persistent_workers
    )

    dl = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, **loader_kwargs)

    out_batches = []
    torch.set_grad_enabled(False)
    # inference-only, no grad / no BN stat updates
    with torch.inference_mode():
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp and use_cuda)
            else nullcontext()
        )
        with amp_ctx:
            for data in _iter_with_rich_progress(dl, desc="Predict proba"):
                # dataset yields a Tensor directly (same as your reference)
                xb = data.to(model.device, non_blocking=use_cuda).float()
                # TabNet forward returns (logits, M_loss); we only need logits
                logits, _ = model.network(xb)
                probs = torch.softmax(logits, dim=1)
                # move to CPU now to keep GPU mem low; convert to numpy at the end
                out_batches.append(probs.detach().cpu())

    out_cpu = torch.cat(out_batches, dim=0)
    return out_cpu.numpy() if return_numpy else out_cpu


def predict_log_proba_fast(
    model,
    X,
    batch_size=None,
    num_workers=None,
    amp=True,
    amp_dtype=torch.float16,
    pin_memory=None,
    prefetch_factor=None,
    persistent_workers=None,
    return_numpy=True,
):
    """
    Fast log-probability inference with the same DataLoader tuning knobs
    as predict_logit_fast/predict_proba_fast.
    """
    model.network.eval()

    if scipy.sparse.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    bs = _resolve_batch_size(model, batch_size)

    loader_kwargs, use_cuda = _resolve_loader_params(
        model.device, num_workers, pin_memory, prefetch_factor, persistent_workers
    )

    dl = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, **loader_kwargs)

    out_batches = []
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp and use_cuda)
            else nullcontext()
        )
        with amp_ctx:
            for data in _iter_with_rich_progress(dl, desc="Predict log proba"):
                xb = data.to(model.device, non_blocking=use_cuda).float()
                logits, _ = model.network(xb)
                log_probs = torch.log_softmax(logits, dim=1)
                out_batches.append(log_probs.detach().cpu())

    out_cpu = torch.cat(out_batches, dim=0)
    return out_cpu.numpy() if return_numpy else out_cpu


def predict_logit_fast(
    model,
    X,
    batch_size=None,
    num_workers=None,
    amp=True,
    amp_dtype=torch.float16,
    pin_memory=None,
    prefetch_factor=None,
    persistent_workers=None,
    return_numpy=True,
):

    # --- match original behavior ---
    model.network.eval()

    # DataLoader construction (reuse the same datasets as original)
    if scipy.sparse.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    bs = _resolve_batch_size(model, batch_size)

    loader_kwargs, use_cuda = _resolve_loader_params(
        model.device, num_workers, pin_memory, prefetch_factor, persistent_workers
    )

    dl = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, **loader_kwargs)

    out_batches = []
    torch.set_grad_enabled(False)
    # inference-only, no grad / no BN stat updates
    with torch.inference_mode():
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp and use_cuda)
            else nullcontext()
        )
        with amp_ctx:
            for data in _iter_with_rich_progress(dl, desc="Predict logit"):
                # dataset yields a Tensor directly (same as your reference)
                xb = data.to(model.device, non_blocking=use_cuda).float()
                # TabNet forward returns (logits, M_loss); we only need logits
                logits, _ = model.network(xb)
                probs = logits
                # move to CPU now to keep GPU mem low; convert to numpy at the end
                out_batches.append(probs.detach().cpu())

    out_cpu = torch.cat(out_batches, dim=0)
    return out_cpu.numpy() if return_numpy else out_cpu

def explain_fast(
    model,
    X,
    normalize: bool = False,
    num_workers: int | None = None,
    desc="Explaining",
    batch_size=8192,
):
    device = model.device  # 이미 TabModel에 있음

    if sp.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    if num_workers is None:
        try:
            import os
            nw = max(2, (os.cpu_count() or 4) // 2)
        except Exception:
            nw = 2
    else:
        nw = num_workers

    pin = device.type == "cuda"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin
    )

    model.network.eval()
    # post_embed_dim x input_dim
    R = _ensure_reducing_t(model, device)
    # 뒤에서 (input_dim x post_embed_dim)로 쓰기 위해 transpose
    R_t = R.transpose(0, 1)

    res_explain_parts = []
    res_masks_lists = None

    eps = 1e-12

    with torch.inference_mode():
        for data in _iter_with_rich_progress(loader, desc=desc):
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device, non_blocking=True).float()

            # 여기서 내부적으로 카테고리 임베딩 다 거친다
            M_explain, masks = model.network.forward_masks(data)
            # M_explain: (batch, post_embed_dim)

            # (batch, post_embed_dim) @ (post_embed_dim, input_dim)를
            # sparse.mm을 쓰기 위해 transpose 트릭:
            # M @ R = (R^T @ M^T)^T
            M_red = torch.sparse.mm(R_t, M_explain.transpose(0, 1)).transpose(0, 1)
            # M_red: (batch, input_dim)

            if normalize:
                s = M_red.sum(dim=1, keepdim=True).clamp_min(eps)
                M_red = M_red / s

            res_explain_parts.append(M_red.cpu().numpy())

            # masks들도 동일하게 reduce
            proc_masks = {}
            for k, v in masks.items():
                # v: (batch, post_embed_dim)
                v_red = torch.sparse.mm(R_t, v.transpose(0, 1)).transpose(0, 1)
                if normalize:
                    sv = v_red.sum(dim=1, keepdim=True).clamp_min(eps)
                    v_red = v_red / sv
                proc_masks[k] = v_red.cpu().numpy()

            if res_masks_lists is None:
                res_masks_lists = {k: [proc_masks[k]] for k in proc_masks.keys()}
            else:
                for k in proc_masks.keys():
                    res_masks_lists[k].append(proc_masks[k])

    res_explain = np.concatenate(res_explain_parts, axis=0)
    res_masks = {k: np.concatenate(vlist, axis=0) for k, vlist in res_masks_lists.items()}

    return res_explain, res_masks

def predict_log_proba(model, X):
    """
    Make predictions for classification on a batch (valid)

    Parameters
    ----------
    X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
        Input data

    Returns
    -------
    res : np.ndarray

    """
    model.network.eval()

    if scipy.sparse.issparse(X):
        dataloader = DataLoader(
            SparsePredictDataset(X),
            batch_size=model.batch_size,
            shuffle=False,
        )
    else:
        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=model.batch_size,
            shuffle=False,
        )

    results = []
    for batch_nb, data in enumerate(dataloader):
        data = data.to(model.device).float()

        output, M_loss = model.network(data)
        predictions = torch.nn.LogSoftmax(dim=1)(output).cpu().detach().numpy()
        results.append(predictions)
    res = np.vstack(results)
    return res


def predict_logit(model, X):
    """
    Make predictions for classification on a batch (valid)

    Parameters
    ----------
    X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
        Input data

    Returns
    -------
    res : np.ndarray

    """
    model.network.eval()

    if scipy.sparse.issparse(X):
        dataloader = DataLoader(
            SparsePredictDataset(X),
            batch_size=model.batch_size,
            shuffle=False,
        )
    else:
        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=model.batch_size,
            shuffle=False,
        )

    results = []
    for batch_nb, data in enumerate(dataloader):
        data = data.to(model.device).float()

        output, M_loss = model.network(data)
        predictions = output.cpu().detach().numpy()
        results.append(predictions)
    res = np.vstack(results)
    return res


class SaveEachEpochCallback(Callback):
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        opt = getattr(self.trainer, "optimizer", None) or getattr(
            self.trainer, "_optimizer", None
        )
        sch = getattr(self.trainer, "scheduler", None) or getattr(
            self.trainer, "_scheduler", None
        )

        if sch is not None and hasattr(sch, "get_last_lr"):
            lrs = sch.get_last_lr()  # 스케줄러 기준(권장)
        elif opt is not None:
            lrs = [g["lr"] for g in opt.param_groups]  # 옵티마이저에서 직접
        else:
            lrs = ["(unknown)"]
        print(f"[epoch {epoch}] lr: {lrs}")
        filename = os.path.join(self.save_dir, f"model_epoch{epoch:03d}")
        # TabNet은 zip archive로 저장됨
        self.trainer.save_model(filename)
        print(f"[Checkpoint] Saved model at {filename}.zip")


def _fnv1a64(s: str) -> np.uint64:
    h = np.uint64(0xCBF29CE484222325)
    for b in s.encode("utf-8", "ignore"):
        h ^= np.uint64(b)
        h *= np.uint64(0x100000001B3)
    return h


def _stable_eids(local_idx: np.ndarray, file_path: str) -> np.ndarray:
    """파일 경로 해시 + 로컬 __idx로부터 전역 안정 EID 생성."""
    salt = _fnv1a64(file_path)
    # 간단 믹스: (salt << 32) ^ local_idx  (local_idx는 uproot의 0..n-1)
    return (salt << np.uint64(32)) ^ local_idx.astype(np.uint64, copy=False)


# fold helper


def _discover_folds(root):
    """
    root 하위에서 data.npz가 존재하고, 폴더명이 fold 패턴을 포함하는 디렉터리를 찾음.
    예: Fold0, fold_1, fold-2 ...
    """
    pat = re.compile(r"(?i)fold[\s_\-]*([0-9]+)")
    folds = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        m = pat.search(name)
        if m:
            folds.append((int(m.group(1)), name, p))
    # fold 인덱스 기준 정렬
    folds.sort(key=lambda x: x[0])
    return folds  # list of (idx, dirname, abspath)


# ---------- helper: pick model zip in a dir ----------
def _pick_zip(model_dir):
    """
    model_dir 안의 .zip 중에서 model_epoch###.zip 패턴이 있으면 가장 큰 epoch 선택,
    없으면 사전순 첫 번째 파일 선택.
    """
    zips = [z for z in os.listdir(model_dir) if z.endswith(".zip")]
    if not zips:
        return None
    best = None
    best_epoch = -1
    for z in zips:
        m = re.search(r"model_epoch(\d+)\.zip$", z)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best = z
    if best is not None:
        return os.path.join(model_dir, best)
    # fallback: lexicographically first
    zips.sort()
    return os.path.join(model_dir, zips[0])


# ----------------------------
# Convenience: fetch one fold's (train/val) arrays
# ----------------------------


def view_fold(dataset: Dict[str, Any], k: int) -> Dict[str, Any]:
    Xtr, ytr, wtr, Xval, yval, wval = get_fold(dataset, k)
    return dict(
        train_features=Xtr,
        train_y=ytr,
        train_weight=wtr,
        val_features=Xval,
        val_y=yval,
        val_weight=wval,
        # 모델 빌드에 그대로 전달
        cat_idxs=dataset["cat_idxs"],
        cat_dims=dataset["cat_dims"],
    )


def get_fold(dataset: Dict[str, np.ndarray], k: int = 0):
    """Return (Xtr, ytr, wtr, Xval, yval, wval) for fold k."""
    X, y, w = dataset["X"], dataset["y"], dataset["weight"]
    folds: List[Tuple[np.ndarray, np.ndarray]] = dataset["folds"]
    if not (0 <= k < len(folds)):
        raise IndexError(f"k={k} out of range for {len(folds)} folds")
    train_idx, val_idx = folds[k]
    return X[train_idx], y[train_idx], w[train_idx], X[val_idx], y[val_idx], w[val_idx]
