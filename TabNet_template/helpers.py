from contextlib import contextmanager, nullcontext
import torch
import numpy as np
from typing import Iterable, Optional, Dict, Any
from time import perf_counter
from pytorch_tabnet.utils import PredictDataset, SparsePredictDataset
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import scipy
import scipy.sparse as sp
from pytorch_tabnet.callbacks import Callback
import os

def _csr_to_torch_sparse_coo(csr: sp.csr_matrix, dtype=np.float32):
    coo = csr.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64)
    values = coo.data.astype(dtype, copy=False)
    i = torch.from_numpy(indices)
    v = torch.from_numpy(values)
    return torch.sparse_coo_tensor(i, v, size=coo.shape, dtype=torch.float32, device="cpu")


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


def compute_class_counts(y: np.ndarray,
                         w: Optional[np.ndarray] = None,
                         num_classes: Optional[int] = None) -> np.ndarray:
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

def predict_proba_fast(model, X,
                       batch_size=None,
                       num_workers=16,
                       amp=True,
                       amp_dtype=torch.float16,
                       pin_memory=True,
                       prefetch_factor=16,
                       persistent_workers=False,
                       return_numpy=True):
    
    # --- match original behavior ---
    model.network.eval()

    # DataLoader construction (reuse the same datasets as original)
    if scipy.sparse.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    bs = int(batch_size) if batch_size is not None else int(model.batch_size)

    use_cuda = isinstance(model.device, torch.device) and model.device.type == "cuda" \
               or (isinstance(model.device, str) and model.device.startswith("cuda"))

    dl = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=(pin_memory and use_cuda),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=False,
    )

    out_batches = []
    torch.set_grad_enabled(False)
    # inference-only, no grad / no BN stat updates
    with torch.inference_mode():
        amp_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if (amp and use_cuda) else nullcontext())
        with amp_ctx:
            for data in tqdm.tqdm(dl):
                # dataset yields a Tensor directly (same as your reference)
                xb = data.to(model.device, non_blocking=use_cuda).float()
                # TabNet forward returns (logits, M_loss); we only need logits
                logits, _ = model.network(xb)
                probs = torch.softmax(logits, dim=1)
                # move to CPU now to keep GPU mem low; convert to numpy at the end
                out_batches.append(probs.detach().cpu())

    out_cpu = torch.cat(out_batches, dim=0)
    return out_cpu.numpy() if return_numpy else out_cpu

def predict_logit_fast(model, X,
                       batch_size=None,
                       num_workers=16,
                       amp=True,
                       amp_dtype=torch.float16,
                       pin_memory=True,
                       prefetch_factor=16,
                       persistent_workers=False,
                       return_numpy=True):
    
    # --- match original behavior ---
    model.network.eval()

    # DataLoader construction (reuse the same datasets as original)
    if scipy.sparse.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    bs = int(batch_size) if batch_size is not None else int(model.batch_size)

    use_cuda = isinstance(model.device, torch.device) and model.device.type == "cuda" \
               or (isinstance(model.device, str) and model.device.startswith("cuda"))

    dl = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=(pin_memory and use_cuda),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=False,
    )

    out_batches = []
    torch.set_grad_enabled(False)
    # inference-only, no grad / no BN stat updates
    with torch.inference_mode():
        amp_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if (amp and use_cuda) else nullcontext())
        with amp_ctx:
            for data in tqdm.tqdm(dl):
                # dataset yields a Tensor directly (same as your reference)
                xb = data.to(model.device, non_blocking=use_cuda).float()
                # TabNet forward returns (logits, M_loss); we only need logits
                logits, _ = model.network(xb)
                probs = logits
                # move to CPU now to keep GPU mem low; convert to numpy at the end
                out_batches.append(probs.detach().cpu())

    out_cpu = torch.cat(out_batches, dim=0)
    return out_cpu.numpy() if return_numpy else out_cpu

def explain_fast(model, X, normalize: bool = False, num_workers: int | None = None, desc="Explaining", batch_size=8192):
    device = torch.device(getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu"))

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=nw, pin_memory=pin)

    model.network.eval()
    R = _ensure_reducing_t(model, device)

    res_explain_parts = []
    res_masks_lists = None

    eps = 1e-12

    with torch.inference_mode():
        for data in tqdm.tqdm(loader, desc=desc):
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device, non_blocking=True).float()

            M_explain, masks = model.network.forward_masks(data)

            M_red = torch.sparse.mm(R, M_explain.transpose(0, 1)).transpose(0, 1)

            if normalize:
                s = M_red.sum(dim=1, keepdim=True).clamp_min(eps)
                M_red = M_red / s

            res_explain_parts.append(M_red.cpu().numpy())

            proc_masks = {}
            for k, v in masks.items():
                v_red = torch.sparse.mm(R, v.transpose(0, 1)).transpose(0, 1)
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
            opt = getattr(self.trainer, "optimizer", None) or getattr(self.trainer, "_optimizer", None)
            sch = getattr(self.trainer, "scheduler", None) or getattr(self.trainer, "_scheduler", None)

            if sch is not None and hasattr(sch, "get_last_lr"):
                lrs = sch.get_last_lr()            # 스케줄러 기준(권장)
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
    h = np.uint64(0xcbf29ce484222325)
    for b in s.encode("utf-8", "ignore"):
        h ^= np.uint64(b)
        h *= np.uint64(0x100000001b3)
    return h

def _stable_eids(local_idx: np.ndarray, file_path: str) -> np.ndarray:
    """파일 경로 해시 + 로컬 __idx로부터 전역 안정 EID 생성."""
    salt = _fnv1a64(file_path)
    # 간단 믹스: (salt << 32) ^ local_idx  (local_idx는 uproot의 0..n-1)
    return ( (salt << np.uint64(32)) ^ local_idx.astype(np.uint64, copy=False) )