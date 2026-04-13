import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

# Allow imports when running from outside the project root
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from helpers import pick_best_device, view_fold
from input_preprocessing import apply_feature_preprocess, fit_preprocess_info
from data.root_data_loader_awk import (
    load_root_as_dataset_kfold as load_data_kfold,
    load_dataset_npz_json,
    save_dataset_npz_json,
)


def _format_list(items: Sequence[str], limit: int = 8) -> str:
    values = [str(item) for item in items if str(item)]
    if not values:
        return "-"
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ... (+{len(values) - limit})"


def _format_winsorize(items: Sequence[tuple[str, tuple[float, float]]], limit: int = 6) -> str:
    values = [f"{name}[{low:g}, {high:g}]" for name, (low, high) in items]
    return _format_list(values, limit=limit)


def _format_counts(counts) -> str:
    if counts is None:
        return "-"
    return ", ".join(f"[{idx}]={float(value):,.1f}" for idx, value in enumerate(counts))


def _preprocess_summary_rows(
    feature_names: Sequence[str],
    categorical_columns: Sequence[str],
    preprocess_info: Dict[str, Any] | None,
) -> list[tuple[str, str]]:
    info = preprocess_info or {}
    rows: list[tuple[str, str]] = []
    used: set[str] = set()
    categorical = list(categorical_columns or [])

    mode = info.get("mode") or "none"
    source = info.get("source") or "none"
    schema = info.get("schema") or "-"
    rows.append(("Mode", f"{mode} ({source}, {schema})"))

    norm_indices = info.get("norm_indices") or []
    if norm_indices:
        rows.append(("Norm rule", "z-score: (x - mean) / std"))
        rows.append(("Norm stats", "fit on current fold train split after winsorize/log1p"))

    group_specs = [
        ("Log", "log_columns", False),
        ("Norm", "norm_columns", False),
        ("Log+Norm", "log_norm_columns", False),
        ("Winsorize", "winsorize_columns", True),
        ("Winsorize+Log", "winsorize_log_columns", True),
        ("Winsorize+Norm", "winsorize_norm_columns", True),
        ("Winsorize+Log+Norm", "winsorize_log_norm_columns", True),
    ]
    for label, key, is_winsorize in group_specs:
        values = info.get(key) or []
        if not values:
            continue
        if is_winsorize:
            rows.append((label, _format_winsorize(values)))
            used.update(name for name, _ in values)
        else:
            rows.append((label, _format_list(values)))
            used.update(values)

    if categorical:
        rows.append(("Categorical", _format_list(categorical)))

    untouched = [
        name for name in feature_names if name not in used and name not in set(categorical)
    ]
    rows.append(("Untouched", _format_list(untouched, limit=10)))
    return rows


def _norm_stats_rows(preprocess_info: Dict[str, Any] | None) -> list[tuple[str, float, float]]:
    info = preprocess_info or {}
    feature_names = list(info.get("feature_names") or [])
    mean = list(info.get("mean") or [])
    scale = list(info.get("scale") or [])
    rows: list[tuple[str, float, float]] = []
    for idx in info.get("norm_indices") or []:
        i = int(idx)
        if i < 0 or i >= len(feature_names) or i >= len(mean) or i >= len(scale):
            continue
        rows.append((str(feature_names[i]), float(mean[i]), float(scale[i])))
    return rows


def _format_optional_float(value) -> str:
    if value is None:
        return "disabled"
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def _print_training_start_summary(
    cfg,
    data,
    data_k,
    preprocess_info,
    model_info,
    train_info,
    counts_train,
    *,
    device,
    out_dir: str,
    fold: int,
    era: str,
    checkpoint,
    pretrained_model,
):
    train_n = int(data_k["train_features"].shape[0])
    val_n = int(data_k["val_features"].shape[0])
    num_features = int(data_k["train_features"].shape[1])
    categorical = list(data.get("cat_columns") or [])
    preprocess_rows = _preprocess_summary_rows(data.get("features") or [], categorical, preprocess_info)
    norm_stats_rows = _norm_stats_rows(preprocess_info)
    init_from = pretrained_model or checkpoint or "scratch"
    class_count = len(counts_train) if counts_train is not None else 0
    train_eval_max = getattr(cfg, "train_eval_max_samples", None)
    report_train_eval = bool(getattr(cfg, "report_train_eval", False))
    val_eval_max = getattr(cfg, "val_eval_max_samples", None)
    sample_bkg = getattr(cfg, "sample_bkg", None)
    epoch_train_event_count = getattr(cfg, "epoch_train_event_count", None)
    epoch_sampling_mode = getattr(cfg, "epoch_sampling_mode", "class_balanced")
    eval_set = list(train_info.get("eval_set") or [])
    val_eval_n = val_n
    if eval_set:
        try:
            val_eval_n = int(eval_set[-1][1].shape[0])
        except Exception:
            val_eval_n = val_n
    if report_train_eval:
        if train_eval_max is None:
            train_eval_text = "full train split"
        else:
            train_eval_text = f"up to {int(train_eval_max):,} train events"
    else:
        train_eval_text = "disabled"
    if val_eval_max is None:
        val_eval_text = "full val split"
    else:
        val_eval_text = f"up to {int(val_eval_max):,} val events"

    try:
        from rich import box
        from rich.columns import Columns
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except Exception:
        print("[training-start]")
        print(
            f"config={cfg.config_name} era={era or '-'} fold={fold} device={device} init={init_from}"
        )
        print(
            f"out={out_dir} train={train_n:,} val={val_n:,} val_eval={val_eval_n:,} "
            f"features={num_features} classes={class_count}"
        )
        print(
            f"batch={train_info['batch_size']:,} vbatch={train_info['virtual_batch_size']:,} max_epochs={train_info['max_epochs']} patience={train_info['patience']}"
        )
        print(
            f"class_balance(sample_bkg)={_format_optional_float(sample_bkg)} epoch_sampler="
            f"{'full train split' if epoch_train_event_count is None else f'{epoch_sampling_mode} / {int(epoch_train_event_count):,}'}"
        )
        print(f"train_eval: {train_eval_text}")
        print(f"val_eval: {val_eval_text} (actual {val_eval_n:,})")
        print(f"class_sumW: {_format_counts(counts_train)}")
        print(
            "Note: TabNet 'loss' includes sparse regularization, while validation "
            "CBFocal metric does not. Compare train/val CBFocal, not loss vs val metric."
        )
        for key, value in preprocess_rows:
            print(f"{key}: {value}")
        if norm_stats_rows:
            print("Norm mean/std:")
            for name, mean_value, std_value in norm_stats_rows:
                print(f"  {name}: mean={mean_value:.6g}, std={std_value:.6g}")
        return

    console = Console(width=140)

    header = Table.grid(padding=(0, 2))
    header.add_column(style="cyan", justify="left")
    header.add_column(justify="left")
    header.add_row("Config", cfg.config_name)
    header.add_row("Era / Fold", f"{era or '-'} / {fold}")
    header.add_row("Device", str(device))
    header.add_row("Init", str(init_from))
    header.add_row("Output", out_dir)

    data_table = Table(title="Data", box=box.SIMPLE_HEAVY)
    data_table.add_column("Item", style="cyan")
    data_table.add_column("Value")
    data_table.add_row("Train events", f"{train_n:,}")
    data_table.add_row("Val events", f"{val_n:,}")
    data_table.add_row("Val eval events", f"{val_eval_n:,}")
    data_table.add_row("Features", f"{num_features}")
    data_table.add_row("Categorical", _format_list(categorical))
    data_table.add_row("Train sumW", f"{float(data_k['train_weight'].sum()):,.1f}")
    data_table.add_row("Val sumW", f"{float(data_k['val_weight'].sum()):,.1f}")

    model_table = Table(title="Model / Train", box=box.SIMPLE_HEAVY)
    model_table.add_column("Item", style="cyan")
    model_table.add_column("Value")
    model_table.add_row("n_d / n_a", f"{cfg.n_d} / {cfg.n_a}")
    model_table.add_row("n_steps", str(cfg.n_steps))
    model_table.add_row("Batch / Virtual", f"{train_info['batch_size']:,} / {train_info['virtual_batch_size']:,}")
    model_table.add_row("Max epochs", str(train_info["max_epochs"]))
    model_table.add_row("Patience", str(train_info["patience"]))
    model_table.add_row("Class balance", f"sample_bkg={_format_optional_float(sample_bkg)}")
    model_table.add_row(
        "Epoch sampler",
        "full train split"
        if epoch_train_event_count is None
        else f"{epoch_sampling_mode} / {int(epoch_train_event_count):,}",
    )
    model_table.add_row("LR / WD", f"{cfg.lr:.3g} / {cfg.weight_decay:.3g}")
    model_table.add_row("Mask", str(cfg.mask_type))
    model_table.add_row("Train eval", train_eval_text)
    model_table.add_row("Val eval", f"{val_eval_text} (actual {val_eval_n:,})")

    class_table = Table(title="Train Class sumW", box=box.SIMPLE_HEAVY)
    class_table.add_column("Class", justify="right", style="cyan")
    class_table.add_column("sumW", justify="right")
    for idx, value in enumerate(counts_train):
        class_table.add_row(str(idx), f"{float(value):,.1f}")

    preprocess_table = Table(title="Preprocess", box=box.SIMPLE_HEAVY)
    preprocess_table.add_column("Group", style="cyan")
    preprocess_table.add_column("Columns", overflow="fold")
    for key, value in preprocess_rows:
        preprocess_table.add_row(key, value)

    console.print(Panel(header, title="Training Start", box=box.SQUARE, padding=(1, 2)))
    console.print(Columns([data_table, model_table, class_table], expand=True))
    console.print(preprocess_table)
    console.print(
        "[yellow]Note:[/yellow] TabNet training [bold]loss[/bold] includes sparse regularization, "
        "while validation [bold]CBFocal[/bold] metric does not. Use train/val CBFocal for a like-for-like gap check."
    )
    if norm_stats_rows:
        norm_table = Table(title="Norm Mean / Std", box=box.SIMPLE_HEAVY)
        norm_table.add_column("Column", style="cyan")
        norm_table.add_column("Mean", justify="right")
        norm_table.add_column("Std", justify="right")
        for name, mean_value, std_value in norm_stats_rows:
            norm_table.add_row(name, f"{mean_value:.6g}", f"{std_value:.6g}")
        console.print(norm_table)


def train(
    model_save_path,
    floss_gamma,
    result_folder_name,
    sample_folder_loc,
    pretrained_model=None,
    checkpoint=None,
    add_year_index=0,
    fold=0,
    TC=None,
    era="",
):
    if TC is None:
        raise ValueError("TC (TrainingConfig module) must be provided.")

    TabNetTrainConfig = TC.TabNetTrainConfig

    # checkpoint와 pretrained를 동시에 쓰지 않기
    if checkpoint is not None and pretrained_model is not None:
        print("하나만 해라")

    fine_tune = pretrained_model is not None
    out_dir = (
        os.path.join(model_save_path, "FineTune") if fine_tune else model_save_path
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = TabNetTrainConfig(
        floss_gamma=floss_gamma,
        fine_tune=fine_tune,
        pretrained_model=pretrained_model,
    )

    # ----- data_info 생성 -----
    data_info = cfg.make_data_info(
        sample_folder_loc=sample_folder_loc,
        result_folder_name=result_folder_name,
        era=era,
        add_year_index=add_year_index,
        sample_bkg=cfg.sample_bkg,
        include_extra_bkgs=False,  # 예전 코드의 break 동작을 그대로 유지
    )

    # ----- 데이터 로딩(+ 캐시) -----
    data_npz = os.path.join(out_dir, "data.npz")
    if os.path.exists(data_npz):
        while True:
            ans = (
                input(
                    f"{data_npz} already exists. Do you want to overwrite it? (y/n): "
                )
                .strip()
                .lower()
            )
            if ans == "y":
                data = load_data_kfold(**data_info)

                save_dataset_npz_json(model_save_path, data)
                break
            elif ans == "n":
                data = load_dataset_npz_json(model_save_path)
                cache_schema = data.get("_cache_meta", {}).get("cache_schema")
                if cache_schema != "raw_features_v3":
                    print(
                        f"Existing cache {data_npz} predates current cache metadata "
                        "(sample_bkg / fold-wise preprocessing). "
                        "Choose 'y' to rebuild it."
                    )
                    continue
                cached_sample_bkg = data.get("_cache_meta", {}).get("sample_bkg")
                if cached_sample_bkg != data_info.get("sample_bkg"):
                    print(
                        f"Existing cache {data_npz} uses sample_bkg={cached_sample_bkg}, "
                        f"but current config requests sample_bkg={data_info.get('sample_bkg')}. "
                        "Choose 'y' to rebuild it."
                    )
                    continue
                print(f"Using existing data {data_npz}")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        data = load_data_kfold(**data_info)
        save_dataset_npz_json(model_save_path, data)

    # ----- 모델/로스/메트릭/트레이닝 인포 -----
    device = pick_best_device(6.0)
    data_k = view_fold(data, fold)
    preprocess_info = fit_preprocess_info(
        data_k["train_features"],
        data["features"],
        mode=data_info.get("preprocess_mode"),
        log_columns=data_info.get("preprocess_log_columns"),
        norm_columns=data_info.get("preprocess_norm_columns"),
        log_norm_columns=data_info.get("preprocess_log_norm_columns"),
        winsorize_columns=data_info.get("preprocess_winsorize_columns"),
        winsorize_log_columns=data_info.get("preprocess_winsorize_log_columns"),
        winsorize_norm_columns=data_info.get("preprocess_winsorize_norm_columns"),
        winsorize_log_norm_columns=data_info.get("preprocess_winsorize_log_norm_columns"),
        categorical_columns=data.get("cat_columns"),
    )
    data_k["train_features"] = apply_feature_preprocess(
        data_k["train_features"], preprocess_info
    )
    data_k["val_features"] = apply_feature_preprocess(
        data_k["val_features"], preprocess_info
    )
    data_k["feature_names"] = list(data.get("features") or [])
    data_k["cat_columns"] = list(data.get("cat_columns") or [])
    data_k["preprocess_info"] = preprocess_info
    data_k["fold"] = int(fold)
    data_k["era"] = str(era or "")
    out_dir = os.path.join(out_dir, f"fold{fold}")
    clf, model_info = cfg.build_model(data_k, device=device, checkpoint=checkpoint)
    loss_fn, eval_metrics, counts_train, _ = cfg.build_loss_and_metrics(
        data_k, device=device
    )
    train_info = cfg.build_train_info(data_k, loss_fn, eval_metrics, out_dir)
    _print_training_start_summary(
        cfg,
        data,
        data_k,
        preprocess_info,
        model_info,
        train_info,
        counts_train,
        device=device,
        out_dir=out_dir,
        fold=fold,
        era=era,
        checkpoint=checkpoint,
        pretrained_model=pretrained_model,
    )

    # ----- 정보 파일 저장(+ 설정 py/json 보존) -----
    cfg.write_info_files(
        model_save_path=out_dir,
        data_info=data_info,
        model_info=model_info,
        train_info=train_info,
        floss_gamma=cfg.floss_gamma,
        counts_train=counts_train,
        preprocess_info=preprocess_info,
        cfg=cfg,  # 현재 설정 인스턴스
        save_config_py=True,  # 이 설정이 정의된 .py 그대로 복사
        save_config_json=True,  # 설정 JSON도 함께 저장
    )

    # ----- 학습/저장/플롯 -----
    clf.fit(**train_info)
    clf.save_model(os.path.join(out_dir, "model"))
