import os
import sys
from pathlib import Path

# Allow imports when running from outside the project root
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from helpers import pick_best_device, view_fold
from data.root_data_loader_awk import (
    load_root_as_dataset_kfold as load_data_kfold,
    load_dataset_npz_json,
    save_dataset_npz_json,
)


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
        sample_bkg=1,
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
    out_dir = os.path.join(out_dir, f"fold{fold}")
    clf, model_info = cfg.build_model(data_k, device=device, checkpoint=checkpoint)
    loss_fn, eval_metrics, counts_train, _ = cfg.build_loss_and_metrics(
        data_k, device=device
    )
    train_info = cfg.build_train_info(data_k, loss_fn, eval_metrics, out_dir)

    # ----- 정보 파일 저장(+ 설정 py/json 보존) -----
    cfg.write_info_files(
        model_save_path=out_dir,
        data_info=data_info,
        model_info=model_info,
        train_info=train_info,
        floss_gamma=cfg.floss_gamma,
        counts_train=counts_train,
        cfg=cfg,  # 현재 설정 인스턴스
        save_config_py=True,  # 이 설정이 정의된 .py 그대로 복사
        save_config_json=True,  # 설정 JSON도 함께 저장
    )

    # ----- 학습/저장/플롯 -----
    clf.fit(**train_info)
    clf.save_model(os.path.join(out_dir, "model"))
