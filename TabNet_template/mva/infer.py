import array
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
import multiprocessing as mp

import numpy as np
import ROOT
from pytorch_tabnet.tab_model import TabNetClassifier

# Ensure local imports work even when executed from another directory
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from helpers import (
    _discover_folds,
    _iter_with_rich_progress,
    _pick_zip,
    predict_log_proba_fast,
    view_fold,
)
from tabnet_compat import load_tabnet_model
from input_preprocessing import apply_feature_preprocess

from data.root_data_loader_awk import load_root_as_dataset_kfold as load_data_kfold


def infer_and_write(root_file, input_model_path, new_branch_name, model_folder, backend="pytorch"):

    folds, model_folds, meta = _load_models_cached(input_model_path, backend=backend)
    if not folds:
        raise RuntimeError(f"No folds found under {input_model_path}.")
    print(
        f"[fold-aware] ({backend}) Found {len(folds)} folds: {[name for _, name, _ in folds]}"
    )

    data_info = meta["data_info"].copy()
    num_class = len(data_info["tree_path_filter_str"])
    data_info["tree_path_filter_str"] = [[(root_file, "Result_Tree", "")]]
    data_info["infer_mode"] = True
    data = load_data_kfold(**data_info)
    if len(model_folds) != len(data["folds"]):
        raise RuntimeError(
            f"Fold count mismatch: model has {len(model_folds)} folds but data has {len(np.unique(data['folds']))} folds"
        )

    result_arr = np.zeros((data["X"].shape[0], num_class), dtype=np.float64)
    all_infer_check = np.zeros(data["X"].shape[0], dtype=bool)

    for idx, name, run_dir in folds:
        data_k = view_fold(data, idx)
        arr = data_k["val_features"]
        runner = model_folds[idx]
        pred = runner.predict_log_proba(arr) if arr.shape[0] > 0 else np.empty((0, num_class))
        pred = np.asarray(pred)

        _, fold_idx = data["folds"][idx]
        all_infer_check[fold_idx] = True
        result_arr[fold_idx] = pred
    if any(~all_infer_check):
        raise RuntimeError(
            f"Some events were not inferred. Check fold assignment and data loading."
        )

    # --- 쓰기 시작 ---
    tf = None
    try:
        tf = ROOT.TFile.Open(root_file, "UPDATE")
        if not tf or tf.IsZombie():
            raise RuntimeError(f"Failed to open for UPDATE: {root_file}")

        tree = tf.Get("Result_Tree")
        if not tree:
            raise RuntimeError(f"'Result_Tree' not found in {root_file}")

        n_entries = tree.GetEntries()
        if n_entries != result_arr.shape[0]:
            raise RuntimeError(
                f"Entry mismatch: tree entries={n_entries} vs pred rows={result_arr.shape[0]} in {root_file}"
            )

        # 브랜치 버퍼/생성
        bufs = [array.array("f", [0.0]) for _ in range(num_class)]
        branches = []
        for cls in range(num_class):
            bname = f"{new_branch_name}_log_prob_{cls}"
            branches.append(tree.Branch(bname, bufs[cls], f"{bname}/F"))

        # 채우기
        for i in range(n_entries):
            for cls in range(num_class):
                bufs[cls][0] = float(result_arr[i, cls])
                branches[cls].Fill()

        # 쓰기
        tree.Write("", ROOT.TObject.kOverwrite)

        # 성공 시 메시지(선택)
        return f"OK: {root_file}"

    finally:
        if tf:
            tf.Close()


def _shutdown_pool(pool, label: str, timeout: float = 15.0):
    """
    Terminate a multiprocessing pool quickly so the main process can exit.
    We avoid the blocking Pool.join() to prevent hangs seen after inference.
    """
    try:
        pool.close()
    except Exception:
        pass

    try:
        pool.terminate()
    except Exception:
        pass

    deadline = time.time() + timeout
    for proc in getattr(pool, "_pool", []):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            proc.join(timeout=remaining)
        except Exception:
            pass

    alive = [p for p in getattr(pool, "_pool", []) if p.is_alive()]
    if alive:
        print(f"[warn] forcing {len(alive)} stuck worker(s) to exit for {label}")
        for proc in alive:
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass


def safe_infer(args):
    f, input_model_path, new_branch_name, model_folder, backend = args
    logs = []

    def log(msg):
        logs.append(msg)

    try:
        log(f"[{os.getpid()}] start {f}")
        msg = infer_and_write(
            f, input_model_path, new_branch_name, model_folder, backend=backend
        )
        log(f"[{os.getpid()}] done  {f}: {msg}")
        return ("ok", f, logs)
    except Exception:
        import traceback

        logs.append(traceback.format_exc())
        return ("err", f, logs)


def infer(input_root_file, input_model_path, branch_name="template_score", backend="pytorch"):
    import array, shutil, time, tempfile
    from pathlib import Path
    import multiprocessing as mp  # torch.multiprocessing가 꼭 필요 없으면 표준 mp가 덜 까다롭습니다.

    start_time = time.time()
    input_root_file = str(Path(input_root_file).resolve())
    print(input_model_path)

    model_folder = str(Path(input_model_path).parent)
    outname = "_".join(Path(input_root_file).with_suffix("").parts[-5:])
    backend = _normalize_backend(backend)
    print(f"[infer] backend={backend}")

    # spawn 설정은 여기서 중복 호출시 에러 -> 무시
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    tmp_dir = Path(input_root_file).parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)  # 없으면 생성

    tmp_root = Path(tempfile.mkdtemp(prefix=f"tmp_infer_{outname}_", dir=tmp_dir))
    print(f"[tmp] working dir: {tmp_root}")

    try:
        new_branch_name = branch_name

        print(f"Start to process {input_root_file}")
        input_file = ROOT.TFile.Open(input_root_file, "READ")
        if not input_file or input_file.IsZombie():
            raise RuntimeError(f"Failed to open: {input_root_file}")
        print(f"Opened {input_root_file}")

        output_files = []

        # 1) 상위 디렉터리 순회
        for ch_key in input_file.GetListOfKeys():
            ch_obj = ch_key.ReadObj()
            if not ch_obj.InheritsFrom("TDirectory"):
                continue

            chdirname = tmp_root / ch_obj.GetName()
            print(f"Will create a new directory: {chdirname}")
            chdirname.mkdir(parents=True, exist_ok=True)

            # 2) 하위 디렉터리 순회
            for key in ch_obj.GetListOfKeys():
                obj = key.ReadObj()
                if not obj.InheritsFrom("TDirectory"):
                    continue

                out_path = chdirname / f"{obj.GetName()}.root"
                output_file = ROOT.TFile.Open(str(out_path), "RECREATE")
                output_files.append(str(out_path))

                # 입력 디렉토리로 cd
                input_file.cd(f"{ch_obj.GetName()}/{obj.GetName()}")

                for inner_key in ROOT.gDirectory.GetListOfKeys():
                    inner_obj = inner_key.ReadObj()

                    output_file.cd()
                    prefix = str(new_branch_name)

                    if inner_obj.InheritsFrom("TTree"):
                        tree = inner_obj
                        # 어떤 브랜치가 지워질지 미리 로깅
                        branches = tree.GetListOfBranches()
                        names = [
                            branches.At(i).GetName() for i in range(branches.GetSize())
                        ]
                        matches = [n for n in names if n.startswith(prefix)]

                        # 예외가 나도 상태 원복되도록
                        tree.SetBranchStatus("*", 1)
                        try:
                            if matches:
                                print(
                                    f"[{ch_obj.GetName()}/{obj.GetName()}]"
                                    f" Tree '{tree.GetName()}': delete {len(matches)} branches with prefix '{prefix}': {matches}"
                                )
                                tree.SetBranchStatus(prefix + "*", 0)
                            cloned = tree.CloneTree(-1, "fast")
                            cloned.SetName(tree.GetName())
                            cloned.Write("", ROOT.TObject.kOverwrite)
                        finally:
                            tree.SetBranchStatus("*", 1)
                    else:
                        inner_obj.Write(inner_obj.GetName(), ROOT.TObject.kOverwrite)

                output_file.Close()

        # 입력 파일은 더 이상 사용 안 하므로 즉시 닫기
        input_file.Close()

        # 3) 멀티프로세싱 추론
        errors, success = [], []

        args = [
            (f, input_model_path, new_branch_name, model_folder, backend)
            for f in output_files
        ]

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=1)
        try:
            for status, f, logs in pool.imap_unordered(safe_infer, args, chunksize=1):
                for line in logs:
                    print(line, flush=True)
                if status == "ok":
                    success.append((f, "\n".join(logs)))
                else:
                    errors.append((f, "\n".join(logs)))
        finally:
            _shutdown_pool(pool, label="infer")

        if errors:
            print("Errors encountered during processing:")
            for f, tb in errors:
                print(f"[{f}]")
                print(tb)
            print("[ABORT] Keeping original ROOT file intact; not merging/replacing.")
            return

        if success:
            print("Successfully processed:")
            for f, msg in success:
                print(f"- {f} {('→ ' + msg) if msg else ''}")  # msg가 list
        # 4) 병합
        merged_path = str(
            Path(input_root_file).with_name(Path(input_root_file).stem + "_merged.root")
        )
        merged_file = ROOT.TFile.Open(merged_path, "RECREATE")
        if not merged_file or merged_file.IsZombie():
            raise RuntimeError(f"Failed to create merged file: {merged_path}")

        for f in output_files:
            src = ROOT.TFile.Open(f, "READ")
            if not src or src.IsZombie():
                raise RuntimeError(f"Failed to open split file: {f}")
            # chdir/dir명은 파일 경로로부터
            chdir = Path(f).parent.name
            filedir = Path(f).stem

            src.cd()
            for inner_key in ROOT.gDirectory.GetListOfKeys():
                inner_obj = inner_key.ReadObj()
                merged_file.cd()
                # 중첩 디렉토리 확보
                full_dir = f"{chdir}/{filedir}"
                d = merged_file.GetDirectory(full_dir)
                if not d:
                    d = merged_file.mkdir(full_dir)
                d.cd()
                if inner_obj.InheritsFrom("TTree"):
                    inner_obj.SetBranchStatus("*", 1)
                    cloned_tree = inner_obj.CloneTree(-1, "fast")
                    cloned_tree.Write()
                else:
                    inner_obj.Write()
            src.Close()

        merged_file.Close()

        # 5) 원본 교체 + 정리
        shutil.move(merged_path, input_root_file)

        end_time = time.time()
        print(f"Elapsed Time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        raise

# parallel_score_friend.py
ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT(0)

# ==== 기존 유틸이 있다고 가정 ====
# TabNetClassifier, _discover_folds, _pick_zip, load_data_kfold, view_fold, predict_log_proba_fast

_MODEL_CACHE = {}

_BACKEND_ALIASES = {
    "torch": "pytorch",
    "pt": "pytorch",
    "pytorch": "pytorch",
    "onnx": "onnx",
    "onnx-int8": "onnx-int8",
    "onnx_int8": "onnx-int8",
    "trt": "tensorrt",
    "trt-int8": "tensorrt-int8",
    "trt_int8": "tensorrt-int8",
    "tensorrt": "tensorrt",
    "tensorrt-int8": "tensorrt-int8",
    "tensorrt_int8": "tensorrt-int8",
}


def _normalize_backend(name: Optional[str]) -> str:
    backend = "pytorch" if not name else _BACKEND_ALIASES.get(name.lower(), name.lower())
    valid = {"pytorch", "onnx", "onnx-int8", "tensorrt", "tensorrt-int8"}
    if backend not in valid:
        raise ValueError(f"Unsupported backend '{name}'. Choose from {sorted(valid)}")
    return backend


def _prefer_int8(backend: str) -> bool:
    return "int8" in backend


def _dl_workers_for_infer() -> int:
    env_v = os.getenv("TABNET_INFER_DL_WORKERS")
    if env_v is not None:
        try:
            return max(0, int(env_v))
        except Exception:
            pass
    try:
        import multiprocessing as omp

        if omp.current_process().daemon:
            return 0  # daemonic process cannot spawn DataLoader workers
    except Exception:
        pass
    return 0  # safe default


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        # keep shape-compatible empty
        if logits.ndim == 1:
            return logits.reshape(0, 0)
        return logits.reshape(logits.shape[0], logits.shape[1] if logits.ndim > 1 else 0)
    maxv = np.max(logits, axis=1, keepdims=True)
    stable = logits - maxv
    sumexp = np.exp(stable).sum(axis=1, keepdims=True)
    return stable - np.log(sumexp)


def _batch_size_from_env(keys: tuple[str, ...], default: int) -> int:
    for k in keys:
        v = os.getenv(k)
        if v is None:
            continue
        try:
            return max(1, int(v))
        except Exception:
            pass
    return default


def _iter_batches(arr: np.ndarray, batch_size: int):
    if batch_size is None or batch_size <= 0:
        yield arr
        return
    n = arr.shape[0]
    for start in range(0, n, batch_size):
        yield arr[start : min(n, start + batch_size)]


def _resolve_onnx_path(model_root: str, fold_idx: int, prefer_int8: bool = False) -> str:
    root = Path(model_root)
    onnx_dir = root / "onnx"
    base = f"tabnet_fold{fold_idx}"
    candidates = []
    if prefer_int8:
        candidates.extend(
            [
                onnx_dir / f"{base}.int8.onnx",
                onnx_dir / f"{base}_int8.onnx",
                root / f"{base}.int8.onnx",
            ]
        )
    candidates.extend(
        [
            onnx_dir / f"{base}.onnx",
            root / f"{base}.onnx",
        ]
    )
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"ONNX model for fold {fold_idx} not found. Looked for: {', '.join(map(str, candidates))}"
    )


def _resolve_trt_plan_path(model_root: str, fold_idx: int, prefer_int8: bool = False) -> Optional[str]:
    root = Path(model_root)
    trt_dir = root / "onnx"
    base = f"tabnet_fold{fold_idx}"
    candidates = []
    if prefer_int8:
        candidates.extend(
            [
                trt_dir / f"{base}.int8.plan",
                trt_dir / f"{base}_int8.plan",
            ]
        )
    candidates.extend([trt_dir / f"{base}.plan", root / f"{base}.plan"])
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _embedded_preprocess_from_metadata(model_meta) -> bool:
    if model_meta is None:
        return False
    custom = getattr(model_meta, "custom_metadata_map", None) or {}
    flag = str(custom.get("tabnet_preprocess_embedded", "0")).strip().lower()
    return flag in {"1", "true", "yes"}


def _onnx_has_embedded_preprocess(onnx_path: Optional[str]) -> bool:
    if not onnx_path:
        return False
    try:
        import onnxruntime as ort
    except Exception:
        return False
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception:
        return False
    return _embedded_preprocess_from_metadata(session.get_modelmeta())


class _TorchFoldRunner:
    def __init__(self, model: TabNetClassifier, preprocess_info=None):
        self.model = model
        self.preprocess_info = preprocess_info

    def predict_log_proba(self, arr: np.ndarray) -> np.ndarray:
        arr = apply_feature_preprocess(arr, self.preprocess_info)
        return predict_log_proba_fast(
            self.model, arr, num_workers=_dl_workers_for_infer()
        )


class _OnnxFoldRunner:
    def __init__(
        self,
        onnx_path: str,
        prefer_trt_provider: bool = False,
        batch_size: Optional[int] = None,
        external_preprocess_info=None,
    ):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise ImportError(
                f"onnxruntime is required for ONNX/TensorRT backends: {exc}"
            )

        available = ort.get_available_providers()
        provider_order = []
        if prefer_trt_provider:
            provider_order.append("TensorrtExecutionProvider")
        provider_order.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        providers = [p for p in provider_order if p in available] or available

        so = ort.SessionOptions()
        if "TensorrtExecutionProvider" in providers:
            cache_dir = Path(onnx_path).parent / "trt_cache"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                so.add_session_config_entry("trt_engine_cache_enable", "1")
                so.add_session_config_entry("trt_engine_cache_path", str(cache_dir))
            except Exception:
                pass

        self.session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=so
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        embedded_preprocess = _embedded_preprocess_from_metadata(
            self.session.get_modelmeta()
        )
        self.preprocess_info = None if embedded_preprocess else external_preprocess_info
        self.batch_size = batch_size or _batch_size_from_env(
            ("ONNX_BATCH_INFER", "BATCH_INFER"), 4096
        )
        print(
            f"[onnx] Loaded {onnx_path} with providers={providers}, "
            f"embedded_preprocess={embedded_preprocess}"
        )

    def predict_log_proba(self, arr: np.ndarray) -> np.ndarray:
        arr = apply_feature_preprocess(arr, self.preprocess_info)
        arr = np.asarray(arr, dtype=np.float32, order="C")
        n = arr.shape[0]
        if n == 0:
            return np.empty((0, arr.shape[1] if arr.ndim > 1 else 0), dtype=np.float64)

        outputs = []
        for start in _iter_with_rich_progress(
            range(0, n, self.batch_size), desc="Predict log proba (onnx)"
        ):
            end = min(n, start + self.batch_size)
            chunk = arr[start:end]
            logits = self.session.run(
                [self.output_name], {self.input_name: chunk}
            )[0]
            outputs.append(_log_softmax_np(logits))

        return np.concatenate(outputs, axis=0) if outputs else np.empty(
            (0, arr.shape[1] if arr.ndim > 1 else 0), dtype=np.float64
        )


class _TrtPlanFoldRunner:
    def __init__(
        self,
        plan_path: str,
        batch_size: Optional[int] = None,
        external_preprocess_info=None,
    ):
        try:
            import tensorrt as trt  # type: ignore
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # type: ignore  # noqa: F401
        except Exception as exc:
            raise ImportError(f"TensorRT plan backend requires tensorrt + pycuda: {exc}")

        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(plan_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine at {plan_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.batch_size = batch_size or _batch_size_from_env(
            ("TRT_BATCH_INFER", "ONNX_BATCH_INFER", "BATCH_INFER"), 1024
        )

        self.trt = trt
        self.cuda = cuda
        self._use_binding_api = hasattr(self.engine, "num_bindings")
        self.input_binding = None
        self.output_binding = None
        self.input_name = None
        self.output_name = None
        self.input_dtype = None
        self.output_dtype = None
        self._output_bindings: list[tuple[int, str]] = []
        self._output_names_all: list[str] = []
        self._output_dtypes: dict[str, np.dtype] = {}

        def _pick_output_name(names: list[str]) -> str:
            lower = [n.lower() for n in names]
            for target in ("logit", "prob"):
                for name, lname in zip(names, lower):
                    if target in lname:
                        return name
            return names[0]

        if self._use_binding_api:
            for i in range(self.engine.num_bindings):
                bname = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.input_name = bname
                else:
                    self._output_bindings.append((i, bname))
            if self.input_binding is None:
                raise RuntimeError("TensorRT plan must have at least one input.")
            if not self._output_bindings:
                raise RuntimeError("TensorRT plan must have at least one output.")
            if len(self._output_bindings) == 1:
                self.output_binding, self.output_name = self._output_bindings[0]
            else:
                chosen = _pick_output_name([n for _, n in self._output_bindings])
                self.output_binding, self.output_name = next((i, n) for i, n in self._output_bindings if n == chosen)
            self.input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_binding)))
            self.output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_binding)))
        else:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1:
                raise RuntimeError(f"TensorRT plan must have exactly one input tensor. inputs={inputs}")
            if not outputs:
                raise RuntimeError("TensorRT plan must have at least one output tensor.")
            self.input_name = inputs[0]
            self.output_name = _pick_output_name(outputs)
            self._output_names_all = outputs
            self._output_dtypes = {
                name: np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))) for name in outputs
            }
            self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
            self.output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        self._num_classes = None
        self.preprocess_info = external_preprocess_info
        print(f"[trt] Loaded TensorRT plan from {plan_path}")

    def predict_log_proba(self, arr: np.ndarray) -> np.ndarray:
        cuda = self.cuda
        trt = self.trt
        arr = apply_feature_preprocess(arr, self.preprocess_info)
        arr = np.asarray(arr, dtype=self.input_dtype, order="C")
        n = arr.shape[0]
        if n == 0:
            n_class = self._num_classes or (arr.shape[1] if arr.ndim > 1 else 0)
            return np.empty((0, n_class), dtype=np.float64)

        outputs = []
        for start in _iter_with_rich_progress(
            range(0, n, self.batch_size), desc="Predict log proba (trt)"
        ):
            end = min(n, start + self.batch_size)
            chunk = arr[start:end]

            if self._use_binding_api:
                self.context.set_binding_shape(self.input_binding, tuple(chunk.shape))
                out_host = None
                bindings = [None] * self.engine.num_bindings
                # input binding
                d_in = cuda.mem_alloc(chunk.nbytes)
                bindings[self.input_binding] = int(d_in)

                # outputs: allocate per-output device buffer
                d_out_primary = None
                out_host = None
                for ob_idx, ob_name in self._output_bindings:
                    shape = tuple(
                        (chunk.shape[0] if dim == -1 else dim)
                        for dim in self.context.get_binding_shape(ob_idx)
                    )
                    dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(ob_idx)))
                    d_buf = cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
                    bindings[ob_idx] = int(d_buf)
                    if ob_idx == self.output_binding:
                        d_out_primary = d_buf
                        out_host = np.empty(shape, dtype=self.output_dtype)
                    else:
                        # store to free later
                        pass

                cuda.memcpy_htod_async(d_in, chunk, self.stream)
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                if d_out_primary is not None:
                    cuda.memcpy_dtoh_async(out_host, d_out_primary, self.stream)
                self.stream.synchronize()

                # free allocations
                d_in.free()
                for ptr in bindings:
                    if ptr is None or ptr == int(d_in):
                        continue
                    try:
                        cuda.mem_free(ptr)
                    except Exception:
                        pass
            else:
                self.context.set_input_shape(self.input_name, tuple(chunk.shape))
                out_shapes = {}
                d_outputs = {}
                out_host = None

                for name in self._output_names_all:
                    shape = tuple(
                        (chunk.shape[0] if dim == -1 else dim)
                        for dim in self.context.get_tensor_shape(name)
                    )
                    out_shapes[name] = shape
                    dtype = self._output_dtypes.get(name, self.output_dtype)
                    d_buf = cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
                    d_outputs[name] = (d_buf, dtype)
                    self.context.set_tensor_address(name, int(d_buf))

                d_in = cuda.mem_alloc(chunk.nbytes)
                self.context.set_tensor_address(self.input_name, int(d_in))

                cuda.memcpy_htod_async(d_in, chunk, self.stream)
                self.context.execute_async_v3(stream_handle=self.stream.handle)

                if self.output_name in d_outputs:
                    out_buf, _dtype = d_outputs[self.output_name]
                    out_host = np.empty(out_shapes[self.output_name], dtype=self.output_dtype)
                    cuda.memcpy_dtoh_async(out_host, out_buf, self.stream)
                self.stream.synchronize()

                d_in.free()
                for buf, _dtype in d_outputs.values():
                    try:
                        buf.free()
                    except Exception:
                        pass

            if self._num_classes is None and out_host.ndim > 1:
                self._num_classes = out_host.shape[1]
            outputs.append(_log_softmax_np(out_host))

        if outputs:
            return np.concatenate(outputs, axis=0)
        # graceful empty result
        n_class = self._num_classes or (arr.shape[1] if arr.ndim > 1 else 0)
        return np.empty((0, n_class), dtype=np.float64)


def _maybe_create_trt_plan_runner(
    plan_path: Optional[str], *, external_preprocess_info=None
) -> Optional[_TrtPlanFoldRunner]:
    if plan_path is None:
        return None
    try:
        return _TrtPlanFoldRunner(
            plan_path, external_preprocess_info=external_preprocess_info
        )
    except Exception as exc:
        print(f"[trt] Failed to use plan {plan_path}: {exc}")
        return None


def _load_fold_meta(run_dir: str) -> dict:
    info_path = os.path.join(run_dir, "info.npy")
    if not os.path.exists(info_path):
        return {}
    return np.load(info_path, allow_pickle=True)[()]


def _load_models_cached(input_model_path: str, backend: str = "pytorch"):
    backend = _normalize_backend(backend)
    cache_key = (backend, input_model_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    folds = _discover_folds(input_model_path)
    if not folds:
        raise RuntimeError(f"No folds under {input_model_path}")

    model_folds = {}
    meta = None
    prefer_int8 = _prefer_int8(backend)
    for idx, name, run_dir in folds:
        fold_meta = _load_fold_meta(run_dir)
        preprocess_info = fold_meta.get("preprocess_info")
        if meta is None:
            meta = fold_meta

        if backend == "pytorch":
            model = TabNetClassifier()
            z = _pick_zip(run_dir)
            if z is None:
                raise FileNotFoundError(f"No model .zip in {run_dir}")
            load_tabnet_model(model, z)
            model_folds[idx] = _TorchFoldRunner(model, preprocess_info=preprocess_info)
        elif backend.startswith("onnx"):
            onnx_path = _resolve_onnx_path(
                input_model_path, idx, prefer_int8=prefer_int8
            )
            model_folds[idx] = _OnnxFoldRunner(
                onnx_path,
                prefer_trt_provider=False,
                external_preprocess_info=preprocess_info,
            )
        elif backend.startswith("tensorrt"):
            plan_path = _resolve_trt_plan_path(
                input_model_path, idx, prefer_int8=prefer_int8
            )
            onnx_path = None
            try:
                onnx_path = _resolve_onnx_path(
                    input_model_path, idx, prefer_int8=prefer_int8
                )
            except FileNotFoundError:
                pass
            external_preprocess_info = (
                None if _onnx_has_embedded_preprocess(onnx_path) else preprocess_info
            )
            runner = _maybe_create_trt_plan_runner(
                plan_path, external_preprocess_info=external_preprocess_info
            )
            if runner is None:
                if onnx_path is None:
                    raise FileNotFoundError(
                        f"No ONNX model found for fold {idx} under {input_model_path}"
                    )
                runner = _OnnxFoldRunner(
                    onnx_path,
                    prefer_trt_provider=True,
                    external_preprocess_info=preprocess_info,
                )
            model_folds[idx] = runner
        else:
            raise ValueError(f"Unknown backend: {backend}")

    if meta is None:
        meta = {}
    _MODEL_CACHE[cache_key] = (folds, model_folds, meta)
    return _MODEL_CACHE[cache_key]

def _iter_subdirs_with_result_tree(in_root: str, tree_name="Result_Tree"):
    f = ROOT.TFile.Open(in_root, "READ");  assert f and not f.IsZombie()
    pairs = []
    tree_name = tree_name.lstrip("/")
    for k in f.GetListOfKeys():
        obj = k.ReadObj()
        if not obj.InheritsFrom("TDirectory"): continue
        ch = obj.GetName()
        for k2 in obj.GetListOfKeys():
            obj2 = k2.ReadObj()
            if not obj2.InheritsFrom("TDirectory"): continue
            sub = obj2.GetName()
            f.cd(f"{ch}/{sub}")
            t = ROOT.gDirectory.Get(tree_name)
            if t and t.InheritsFrom("TTree"):
                pairs.append((ch, sub))
    if not pairs:
        tree = f.Get(tree_name)
        if tree and tree.InheritsFrom("TTree"):
            pairs.append(("", ""))
    f.Close()
    return pairs


def _list_root_level_trees(in_root: str):
    """Return names of TTrees at the root level of the file."""
    f = ROOT.TFile.Open(in_root, "READ")
    if not f or f.IsZombie():
        return []
    names = []
    try:
        for key in f.GetListOfKeys():
            obj = key.ReadObj()
            if obj.InheritsFrom("TTree"):
                names.append(obj.GetName())
    finally:
        f.Close()
    names = [name for name in names if name.startswith("El") or name.startswith("Mu")]

    return names



def _infer_logp(
    shard_file: str,
    input_model_path: str,
    tree_name: str = "Result_Tree",
    tree_path: Optional[str] = None,
    backend: str = "pytorch",
) -> np.ndarray:
    folds, model_folds, meta = _load_models_cached(input_model_path, backend=backend)
    data_info = meta["data_info"].copy()
    K = len(data_info["tree_path_filter_str"])
    data_info["infer_mode"] = True
    # tree_path가 주어지면 그대로 사용하고, 없으면 tree_name만 사용
    clean_tree_name = tree_name.lstrip("/")
    use_tree_path = tree_path.lstrip("/") if tree_path else clean_tree_name
    print(f"Loading data from: {shard_file} with tree path: {use_tree_path}")
    data_info["tree_path_filter_str"] = [[(shard_file, use_tree_path, "")]]
    data = load_data_kfold(**data_info)
    N = data["X"].shape[0]
    out = np.zeros((N, K), dtype=np.float64)
    mask = np.zeros(N, dtype=bool)
    for idx, name, _ in folds:
        d = view_fold(data, idx)
        arr = d["val_features"]
        pred = (
            model_folds[idx].predict_log_proba(arr) if arr.shape[0] > 0 else np.empty((0, K))
        )
        pred = np.asarray(pred)
        _, fold_idx = data["folds"][idx]
        mask[fold_idx] = True
        out[fold_idx] = pred
    if not mask.all():
        raise RuntimeError("Incomplete inference coverage.")
    return out  # (N,K)

def _write_logp_tree(dest_file: str, chdir: str, subdir: str, logp: np.ndarray, tree_name="Result_Tree"):
    # dest_file는 미리 RECREATE로 만들어져 있다고 가정
    f = ROOT.TFile.Open(str(dest_file), "UPDATE");  assert f and not f.IsZombie()
    current_dir = f  # root or subdir
    if chdir:
        current_dir = f.GetDirectory(chdir) or f.mkdir(chdir)
    current_dir.cd()
    if subdir:
        current_dir = current_dir.GetDirectory(subdir) or current_dir.mkdir(subdir)
    current_dir.cd()
    t = ROOT.TTree(tree_name, "log_prob_* only")
    N, K = logp.shape
    bufs = [array.array("f", [0.0]) for _ in range(K)]
    for k in range(K):
        b = f"log_prob_{k}"
        t.Branch(b, bufs[k], f"{b}/F")
    for i in range(N):
        row = logp[i]
        for k in range(K):
            bufs[k][0] = float(row[k])
        t.Fill()
    t.Write("", ROOT.TObject.kOverwrite)
    f.Close()


def _merge_parts_to_dest(dest_file: str, parts_dir: str, pairs, tree_name="Result_Tree"):
    return _merge_parts_to_dest_append(dest_file, parts_dir, pairs, tree_name=tree_name, append=False)


def _merge_parts_to_dest_append(dest_file: str, parts_dir: str, pairs, tree_name="Result_Tree", append: bool = False):
    tree_name = tree_name.lstrip("/")
    if not append or not os.path.exists(dest_file):
        ROOT.TFile.Open(dest_file, "RECREATE").Close()
    fout = ROOT.TFile.Open(dest_file, "UPDATE");  assert fout and not fout.IsZombie()
    try:
        for chdir, subdir in pairs:
            p = Path(parts_dir) / f"{chdir}__{subdir}.root"
            if not p.exists():
                raise RuntimeError(f"Missing part: {p}")
            src = ROOT.TFile.Open(str(p), "READ");  assert src and not src.IsZombie()
            if chdir or subdir:
                path = os.path.join(*[q for q in (chdir, subdir) if q])
                src.cd(path)
                t = ROOT.gDirectory.Get(tree_name)
            else:
                t = src.Get(tree_name)
            assert t and t.InheritsFrom("TTree")
            if chdir:
                d1 = fout.GetDirectory(chdir) or fout.mkdir(chdir)
            else:
                d1 = fout
            d1.cd()
            if subdir:
                d2 = d1.GetDirectory(subdir) or d1.mkdir(subdir)
            else:
                d2 = d1
            d2.cd()
            t.CloneTree(-1, "fast").Write(tree_name, ROOT.TObject.kOverwrite)
            src.Close()
            fout.cd()
    finally:
        fout.Close()


def _worker_infer_part(args):
    (
        input_root_file,
        input_model_path,
        part_path,
        chdir,
        subdir,
        tree_name,
        backend,
    ) = args
    tree_path = "/".join([p for p in (chdir, subdir, tree_name) if p])
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        logp = _infer_logp(
            input_root_file,
            input_model_path,
            tree_name=tree_name,
            tree_path=tree_path,
            backend=backend,
        )
        ROOT.TFile.Open(part_path, "RECREATE").Close()
        _write_logp_tree(part_path, chdir, subdir, logp, tree_name=tree_name)
        return ("ok", tree_path, part_path)
    except Exception:
        import traceback

        return ("err", tree_path, traceback.format_exc())


def make_score_friend_file_parallel(input_root_file: str,
                                    input_model_path: str,
                                    out_group_name: str,
                                    tree_name: str = "Result_Tree",
                                    backend: str = "pytorch",
                                    num_workers: int = 2,
                                    cleanup_parts: bool = True,
                                    orig_name = None) -> str:
    """
    원본 옆 out_group_name/ 폴더에 같은 파일명으로 score-only ROOT 생성.
    내부 구조는 원본과 동일(chdir/subdir/Result_Tree) + 브랜치는 log_prob_*.
    트리별로 병렬 추론 → part 파일 → 병합 순서로 동작.
    """
    backend = _normalize_backend(backend)
    input_root_file = str(Path(input_root_file).resolve())
    tree_name = tree_name.lstrip("/")
    base_dir = Path(input_root_file).parent
    src_name = Path(input_root_file).name

    out_dir = base_dir / out_group_name
    out_dir.mkdir(exist_ok=True)
    parts_dir = out_dir / f".parts__{Path(src_name).stem}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    tree_candidates = [tree_name]
    pairs = _iter_subdirs_with_result_tree(input_root_file, tree_name=tree_name)
    if not pairs:
        alt_tree = (
            "Result_Tree"
            if tree_name != "Result_Tree"
            else "Template_Training_Tree"
        )
        if alt_tree != tree_name:
            alt_pairs = _iter_subdirs_with_result_tree(
                input_root_file, tree_name=alt_tree
            )
            if alt_pairs:
                tree_name = alt_tree
                pairs = alt_pairs
                tree_candidates = [tree_name]
    if not pairs:
        detected_list = _list_root_level_trees(input_root_file)
        if detected_list:
            print(f"[multi-tree] detected TTrees at root: {detected_list}")
            tree_candidates = detected_list
        else:
            raise RuntimeError(f"No subdirs with {tree_name} found.")

    # 결과 파일 생성 준비
    dest_file = str(out_dir / (orig_name if orig_name else src_name))
    errs, oks = [], []
    dest_created = False

    # iterate over candidate tree names; useful for 2024 files containing many TTrees
    for tname in tree_candidates:
        pairs = _iter_subdirs_with_result_tree(input_root_file, tree_name=tname)
        if not pairs:
            print(f"[warn] no pairs found for tree '{tname}' in {input_root_file}")
            continue

        safe_tname = tname.replace("/", "__")
        parts_dir_tree = parts_dir / safe_tname
        parts_dir_tree.mkdir(parents=True, exist_ok=True)

        tasks = [
            (
                input_root_file,
                input_model_path,
                str(parts_dir_tree / f"{ch}__{sub}.root"),
                ch,
                sub,
                tname,
                backend,
            )
            for ch, sub in pairs
        ]

        worker_count = max(1, num_workers)
        if worker_count == 1:
            for t in tasks:
                status, tag, payload = _worker_infer_part(t)
                if status == "ok":
                    print(f"[OK ] {tname}:{tag}  → {payload}")
                    oks.append(payload)
                else:
                    print(f"[ERR] {tname}:{tag}\n{payload}")
                    errs.append(f"{tname}:{tag}")
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=worker_count)
            try:
                for status, tag, payload in pool.imap_unordered(_worker_infer_part, tasks, chunksize=1):
                    if status == "ok":
                        print(f"[OK ] {tname}:{tag}  → {payload}")
                        oks.append(payload)
                    else:
                        print(f"[ERR] {tname}:{tag}\n{payload}")
                        errs.append(f"{tname}:{tag}")
            finally:
                _shutdown_pool(pool, label="score-friend")

        if errs:
            raise RuntimeError(f"Part failures: {errs}")

        _merge_parts_to_dest_append(
            dest_file, str(parts_dir_tree), pairs, tree_name=tname, append=dest_created
        )
        dest_created = True

    if cleanup_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    return dest_file

# patch_list는 네가 준 그대로라고 가정
patch_list = {
    "2016preVFP": [
        "QCD_bEnriched_HT2000toInf",
    ],
    "2016postVFP": [
        "DYJets_MG",
        "SingleTop_sch_Lep",
        "SingleTop_tch_antitop_Incl",
        "SingleTop_tch_top_Incl",
        "WZ_pythia",
        "ZZ_pythia",
        "ttZToQQ_ll",
    ],
    "2017": [
        "DYJets_MG",
        "SingleTop_sch_Lep",
        "SingleTop_tch_antitop_Incl",
        "SingleTop_tch_top_Incl",
        "WZ_pythia",
        "ZZ_pythia",
        "ttZToQQ_ll",
    ],
    "2018": [
        "DYJets_MG",
        "SingleTop_sch_Lep",
        "SingleTop_tch_antitop_Incl",
        "SingleTop_tch_top_Incl",
        "WZ_pythia",
        "ZZ_pythia",
        "ttZToQQ_ll",
    ],
}

# --- 루프 위(한 번만)에서 역인덱스 준비 ---
allowed_by_sample = {}
for era, samples in patch_list.items():
    for s in samples:
        allowed_by_sample.setdefault(s, set()).add(era)

# 긴 이름(예: TTLL_powheg_CP5Down) 먼저 매칭되도록 정렬
samples_by_len = sorted(allowed_by_sample.keys(), key=len, reverse=True)


def infer_with_iter(
    input_folder,
    input_model_path,
    branch_name,
    result_folder_name,
    backend="pytorch",
    era=None,
    local=False,
    num_workers=1,
):
    import pathlib
    import shutil
    import ROOT

    if not local:
        import htcondor

    backend = _normalize_backend(backend)
    log_path = None
    if not local:
        log_path = os.path.join(
            os.environ["DIR_PATH"],
            "TabNet_template",
            pathlib.Path(input_model_path).parent.absolute(),
            "infer_log",
        )

    local_log_file = None
    local_log_records = []
    log_started_at = None
    if local:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "infer_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        log_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_branch = branch_name.replace("/", "_")
        local_log_file = log_dir / f"infer_iter_{safe_branch}_{run_tag}.log"
        print(f"[local log] progress will be recorded to {local_log_file}")

    eras = [era] if era != "All" else ["2016preVFP", "2016postVFP", "2017", "2018"]

    def _collect_legacy_entries(e):
        base = os.path.join(input_folder, e, result_folder_name)
        if not os.path.isdir(base):
            return []
        systs = [
            f
            for f in os.listdir(base)
            if os.path.isdir(os.path.join(base, f)) and "." not in f
        ]
        
        entries = []
        for syst in systs:
            syst_dir = os.path.join(base, syst)
            files = [
                os.path.join(syst_dir, f)
                for f in os.listdir(syst_dir)
                if f.endswith(".root")
            ]
            # only keep files "DATA" in their name for data era
            files = [f for f in files if (("4f" in f or "ttH" in f))]
            if files:
                # legacy files don't have a SPANET partner → keep tuple length 3
                entries.extend([(file, syst, None) for file in files])

        
        print(f"[info] entries for {e}: {entries}")
        return entries



    def _collect_2024_entries():
        # --- 1) 후보 디렉토리 수집 -------------------------------------------------
        candidate_dir = Path(input_folder) / Path(result_folder_name)
        yielded: set[str] = set()  # 이미 yield한 절대경로 string

        # 채널은 하드코딩하지 않고, result_folder 아래에서 2024 서브디렉토리를 가진 것만 자동 탐색
        # result_folder_name에 채널 경로까지 포함되면(예: Vcb_SL/Mu_Unmapped_QuadJet_TemplateTraining)
        # 해당 디렉토리 자체를 채널로 취급한다.
        if not candidate_dir.is_dir():
            print(f"[warn] result folder not found: {candidate_dir}")
            return
        if (candidate_dir / "2024").is_dir():
            channel_dirs = [candidate_dir]
        else:
            channel_dirs = sorted(
                d
                for d in candidate_dir.iterdir()
                if d.is_dir() and (d / "2024").is_dir()
            )
        if not channel_dirs:
            print(f"[warn] no 2024 channels found under: {candidate_dir}")
            return

        base_files: dict[str, list[Path]] = {}

        def _gather(base_dir: Path, channel: str, store: dict):
            files = []
            for entry in sorted(os.listdir(base_dir)):
                if not entry.endswith(".root"):
                    continue
                path = base_dir / entry
                files.append(path)
            store[channel] = files

        cand = candidate_dir.resolve()
        for ch_dir in channel_dirs:
            ch = ch_dir.name
            base_dir = ch_dir / "2024"
            spanet_dir = base_dir / "SPANET"
            if base_dir.is_dir():
                _gather(base_dir, ch, base_files)
            else:
                print(f"[warn] missing base dir for 2024: {base_dir}")
            if spanet_dir.is_dir():
                for val in base_files.values():
                    for root_file_abs_path in val:
                        fname = root_file_abs_path.name
                        spanet_path = spanet_dir / fname
                        if spanet_path.is_file():
                            yield (str(root_file_abs_path), "Central", str(spanet_path))
                        else:
                            # SPANET 짝이 없어도 base 파일을 그대로 사용
                            yield (str(root_file_abs_path), "Central", None)
            else:
                # SPANET 디렉토리 자체가 없으면 base 파일만 처리
                for val in base_files.values():
                    for root_file_abs_path in val:
                        yield (str(root_file_abs_path), "Central", None)
                    

        

    def _iter_root_files(directory, label):
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(".root"):
                continue
            yield (os.path.join(directory, entry), label)

    Arg_list_file = "infer_arg_list.txt"
    if not local:
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, Arg_list_file), "w") as f:
            f.write("")

    local_results = []
    for e in eras:
        print(e)
        entries = []
        if e == "2024":
            entries.extend(list(_collect_2024_entries()))
        print(entries)

        entries.extend(_collect_legacy_entries(e))
        if not entries:
            continue


        def _normalize_entry(item):

            if len(item) >= 3:
                return (item[0], item[1], item[2])
            if len(item) == 2:
                return (item[0], item[1], None)
            raise ValueError(f"Unexpected entry format: {item}")

        for entry in entries:
            print(entry)
            temp_cleanup = None
            record = {
                "era": e,
                "syst": None,
                "file": None,
                "spanet_partner": None,
                "target": None,
                "status": "pending",
                "detail": "",
                "output": None,
            }
            try:
                file, syst, spanet_partner = _normalize_entry(entry)
                record.update(
                    {
                        "file": file,
                        "syst": syst,
                        "spanet_partner": spanet_partner,
                        "target": file,
                    }
                )
                if "temp" in file or "update" in file:
                    os.remove(file)
                    record["status"] = "skipped"
                    record["detail"] = "removed temp/update file"
                    continue

                target_file = file

                if spanet_partner:
                    import uproot
                    import awkward as ak

                    tabnet_dir = Path(file).parent / "TABNET"
                    tabnet_dir.mkdir(exist_ok=True)
                    fd, tmp_name = tempfile.mkstemp(
                        prefix=f"{Path(file).stem}__",  # 예: Skim_Vcb_SL_...__
                        suffix="__merged.root",         # 확장자는 .root 유지
                        dir=str(Path(file).parent / "TABNET"),
                    )
                    os.close(fd)
                    out_path = Path(tmp_name)
                    temp_cleanup = out_path

                    with uproot.open(file) as a, uproot.open(spanet_partner) as b:
                        trees_a = {
                            k.split(";")[0]
                            for k, cls in a.classnames().items()
                            if cls == "TTree"
                        }
                        trees_b = {
                            k.split(";")[0]
                            for k, cls in b.classnames().items()
                            if cls == "TTree"
                        }
                        common = sorted(trees_a & trees_b)
                        if not common and (len(trees_a) > 0 or len(trees_b) > 0):
                            raise RuntimeError(
                                f"No common TTrees between {file} and {spanet_partner}"
                            )

                        with uproot.recreate(out_path) as fout:
                            for tname in common:
                                arr_a = a[tname].arrays(library="ak")
                                arr_b = b[tname].arrays(library="ak")
                                if len(arr_a) != len(arr_b):
                                    raise RuntimeError(
                                        f"Entry mismatch for tree {tname}: {len(arr_a)} vs {len(arr_b)}"
                                    )
                                merged = ak.with_name(arr_a, None)
                                for field in arr_b.fields:
                                    if field in merged.fields:
                                        continue
                                    merged[field] = arr_b[field]

                                # 각 브랜치별로 dict로 넘겨야 jagged도 잘 써짐
                                branch_dict = {field: merged[field] for field in merged.fields}
                                fout[tname] = branch_dict

                    print(f"[merge] {syst}: {file} + {spanet_partner} -> {out_path}")
                    target_file = str(out_path)
                    record["target"] = target_file

                print(target_file)
                outname = target_file.split("/")
                outname[-1] = outname[-1].replace(".root", "")
                outname = "_".join(outname[-5:])
                if local:
                    tree_name = "Template_Training_Tree" if e == "2024" else "Result_Tree"
                    print(f"[local infer] era={e} syst={syst} -> {outname}")
                    res_path = make_score_friend_file_parallel(
                        input_root_file=target_file,
                        input_model_path=input_model_path,
                        out_group_name=branch_name,
                        tree_name=tree_name,
                        backend=backend,
                        num_workers=max(1, int(num_workers) if num_workers else 1),
                        orig_name=Path(file).name,
                    )
                    local_results.append((target_file, res_path))
                    record["status"] = "success"
                    record["output"] = res_path
                    record["detail"] = "ok"
                else:
                    with open(os.path.join(log_path, Arg_list_file), "a") as f:
                        f.write(f"{input_model_path} {target_file} {branch_name} {e} {backend}\n")
                    job = htcondor.Submit(
                        {
                            "universe": "vanilla",
                            "getenv": True,
                            "jobbatchname": f"Vcb_infer_{e}_{syst}_{outname}",
                            "executable": str(
                                Path(__file__).resolve().parents[1]
                                / "scripts"
                                / "infer_write.sh"
                            ),
                            "arguments": f"{input_model_path} {target_file} {branch_name} {e} {backend}",
                            "output": os.path.join(log_path, f"{outname}.out"),
                            "error": os.path.join(log_path, f"{outname}.err"),
                            "log": os.path.join(log_path, f"{outname}.log"),
                            "request_memory": (
                                "220GB"
                                if ("TTLJ_powheg" in outname or "TTLL_powheg" in outname)
                                and "Central" in outname
                                else "32GB"
                            ),
                            "request_gpus": (
                                0
                                if ("TTLJ_powheg" in outname and "Central" in outname)
                                else 0
                            ),
                            "request_cpus": 32,
                            "should_transfer_files": "YES",
                            "on_exit_hold": "(ExitBySignal == True) || (ExitCode != 0)",
                        }
                    )

                    schedd = htcondor.Schedd()
                    with schedd.transaction() as txn:
                        cluster_id = job.queue(txn)
                    print("Job submitted with cluster ID:", cluster_id)
                    record["status"] = "queued"
                    record["detail"] = f"cluster {cluster_id}"
            except Exception as exc:
                import traceback

                record["status"] = "fail"
                record["detail"] = traceback.format_exc().strip().replace("\n", " | ")
                print(f"[ERR] inference failed for {entry}: {exc}")
            finally:
                if temp_cleanup:
                    try:
                        os.remove(temp_cleanup)
                    except OSError:
                        pass
                if local:
                    local_log_records.append(record)

    if local and local_log_file:
        try:
            success_cnt = sum(1 for rec in local_log_records if rec["status"] == "success")
            fail_cnt = sum(1 for rec in local_log_records if rec["status"] == "fail")
            skip_cnt = sum(1 for rec in local_log_records if rec["status"] == "skipped")
            with open(local_log_file, "w") as f:
                header = (
                    f"run_start={log_started_at or time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"backend={backend}, branch={branch_name}, "
                    f"model={input_model_path}, workers={num_workers}, era={era}\n"
                )
                f.write(header)
                f.write(f"success={success_cnt}, fail={fail_cnt}, skipped={skip_cnt}\n")
                for rec in local_log_records:
                    base = (
                        f"[{rec['status'].upper():7}] era={rec.get('era')} "
                        f"syst={rec.get('syst')} file={rec.get('file')}"
                    )
                    if rec.get("spanet_partner"):
                        base += f" spanet={rec.get('spanet_partner')}"
                    if rec.get("target") and rec.get("target") != rec.get("file"):
                        base += f" target={rec.get('target')}"
                    if rec["status"] == "success" and rec.get("output"):
                        base += f" -> {rec['output']}"
                    elif rec["status"] == "skipped":
                        base += f" ({rec.get('detail')})"
                    else:
                        base += f" err={rec.get('detail')}"
                    f.write(base + "\n")
            print(
                f"[local log] wrote inference summary to {local_log_file} "
                f"(ok={success_cnt}, fail={fail_cnt}, skip={skip_cnt})"
            )
        except Exception as exc:
            print(f"[warn] failed to write local log {local_log_file}: {exc}")

    if local:
        return local_results
