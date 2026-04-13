from __future__ import annotations

import io
import json
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def _torch_load_bytes(payload: bytes, *, map_location, weights_only: bool | None):
    kwargs = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only

    return torch.load(io.BytesIO(payload), **kwargs)


def _load_state_dict_from_zip(
    model_zip: zipfile.ZipFile,
    *,
    map_location,
):
    with model_zip.open("network.pt") as member:
        payload = member.read()

    try:
        return _torch_load_bytes(
            payload,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return _torch_load_bytes(
            payload,
            map_location=map_location,
            weights_only=None,
        )
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"You are using `torch\.load` with `weights_only=False`.*",
            )
            return _torch_load_bytes(
                payload,
                map_location=map_location,
                weights_only=False,
            )


def load_tabnet_model(model, filepath: str):
    """Load a TabNet zip model without the `torch.load` future warning."""
    try:
        with zipfile.ZipFile(filepath) as model_zip:
            with model_zip.open("model_params.json") as member:
                loaded_params = json.load(member)
                loaded_params["init_params"]["device_name"] = model.device_name
            saved_state_dict = _load_state_dict_from_zip(
                model_zip,
                map_location=model.device,
            )
    except KeyError as exc:
        raise KeyError("Your zip file is missing at least one component") from exc

    model.__init__(**loaded_params["init_params"])
    model._set_network()
    model.network.load_state_dict(saved_state_dict)
    model.network.eval()
    model.load_class_attrs(loaded_params["class_attrs"])
    return model


def _tabnet_input_dim(model) -> int:
    dim = getattr(model, "input_dim", None)
    if dim is None and hasattr(model, "network"):
        dim = getattr(model.network, "input_dim", None)
    if dim is None:
        raise RuntimeError("Could not resolve input_dim from TabNet model.")
    return int(dim)


def build_tabnet_verification_input(
    model,
    *,
    batch_size: int = 64,
    seed: int = 0,
):
    import numpy as np

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    input_dim = _tabnet_input_dim(model)
    cat_idxs = list(getattr(model, "cat_idxs", []) or [])
    cat_dims = list(getattr(model, "cat_dims", []) or [])
    if len(cat_idxs) != len(cat_dims):
        raise RuntimeError(
            f"cat_idxs and cat_dims length mismatch: {len(cat_idxs)} vs {len(cat_dims)}"
        )

    rng = np.random.default_rng(seed)
    batch = rng.normal(size=(batch_size, input_dim)).astype(np.float32)
    batch[0] = 0.0

    for idx, dim in zip(cat_idxs, cat_dims):
        if not 0 <= int(idx) < input_dim:
            raise RuntimeError(
                f"Categorical feature index {idx} is out of range for input_dim={input_dim}"
            )
        if int(dim) <= 0:
            raise RuntimeError(f"Invalid categorical dimension {dim} for feature {idx}")
        batch[:, int(idx)] = rng.integers(0, int(dim), size=batch_size, endpoint=False)
        batch[0, int(idx)] = 0

    return batch


def load_tabnet_verification_input_from_dataset(
    data_root: str | Path,
    *,
    fold_idx: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
):
    import numpy as np

    data_root = Path(data_root)
    data_path = data_root / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    with np.load(data_path, allow_pickle=False) as npz:
        X = np.asarray(npz["X"], dtype=np.float32)
        folds_lab = np.asarray(npz["folds_lab"], dtype=np.int16) if "folds_lab" in npz else None

    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D features in {data_path}, got shape={X.shape}")
    if X.shape[0] == 0:
        raise RuntimeError(f"No rows found in {data_path}")

    candidate = X
    if fold_idx is not None and folds_lab is not None:
        mask = folds_lab == int(fold_idx)
        if mask.any():
            candidate = X[mask]

    rng = np.random.default_rng(seed)
    if candidate.shape[0] >= batch_size:
        indices = rng.choice(candidate.shape[0], size=batch_size, replace=False)
        batch = candidate[indices]
    else:
        indices = rng.choice(candidate.shape[0], size=batch_size, replace=True)
        batch = candidate[indices]

    return np.ascontiguousarray(batch, dtype=np.float32)


def verify_tabnet_onnx_export(
    model,
    onnx_path: str | Path,
    *,
    data_root: str | Path | None = None,
    fold_idx: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
    atol: float = 2e-3,
    rtol: float = 1e-4,
    reference_model=None,
) -> dict[str, Any]:
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX verification. "
            "Install onnxruntime or disable verification."
        ) from exc

    sample_source = "synthetic"
    if data_root is not None:
        try:
            sample = load_tabnet_verification_input_from_dataset(
                data_root,
                fold_idx=fold_idx,
                batch_size=batch_size,
                seed=seed,
            )
            sample_source = "dataset"
        except FileNotFoundError:
            sample = build_tabnet_verification_input(
                model,
                batch_size=batch_size,
                seed=seed,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load verification samples from {data_root}: {exc}"
            ) from exc
    else:
        sample = build_tabnet_verification_input(
            model,
            batch_size=batch_size,
            seed=seed,
        )

    reference_module = reference_model if reference_model is not None else model.network
    if hasattr(reference_module, "eval"):
        reference_module.eval()

    try:
        reference_device = next(reference_module.parameters()).device
    except StopIteration:
        reference_device = getattr(model, "device", torch.device("cpu"))

    sample_tensor = torch.from_numpy(sample).to(device=reference_device)
    with torch.no_grad():
        reference = reference_module(sample_tensor)

    if isinstance(reference, (tuple, list)):
        reference = reference[0]

    reference = reference.detach().cpu().numpy()
    if not np.isfinite(reference).all():
        raise RuntimeError("Reference TabNet output contains non-finite values.")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()
    if len(input_meta) != 1:
        raise RuntimeError(
            f"Expected exactly one ONNX input, found {len(input_meta)} in {onnx_path}"
        )

    exported = session.run(None, {input_meta[0].name: sample})[0]
    if reference.shape != exported.shape:
        raise RuntimeError(
            f"Output shape mismatch for {onnx_path}: "
            f"reference={reference.shape}, exported={exported.shape}"
        )
    if not np.isfinite(exported).all():
        raise RuntimeError(f"Exported ONNX output contains non-finite values: {onnx_path}")

    diff = np.abs(reference - exported)
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    p99_abs_diff = float(np.quantile(diff, 0.99)) if diff.size else 0.0

    if not np.allclose(reference, exported, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"ONNX verification failed for {onnx_path}: "
            f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}, "
            f"p99_abs_diff={p99_abs_diff:.6g}, "
            f"atol={atol}, rtol={rtol}, batch_size={batch_size}, seed={seed}"
        )

    return {
        "batch_size": batch_size,
        "seed": seed,
        "sample_source": sample_source,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "atol": atol,
        "rtol": rtol,
    }


@contextmanager
def suppress_tabnet_onnx_warnings() -> Iterator[None]:
    """Hide known TabNet trace warnings while keeping other export warnings visible."""
    tracer_warning = getattr(torch.jit, "TracerWarning", Warning)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=tracer_warning,
            message=r"Iterating over a tensor might cause the trace to be incorrect\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=tracer_warning,
            message=r"Converting a tensor to a Python boolean might cause the trace to be incorrect\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=tracer_warning,
            message=r"Converting a tensor to a Python integer might cause the trace to be incorrect\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=tracer_warning,
            message=r"Converting a tensor to a NumPy array might cause the trace to be incorrect\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=tracer_warning,
            message=r"torch\.from_numpy results are registered as constants in the trace\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r"__array_wrap__ must accept context and return_scalar arguments .*",
        )
        yield
