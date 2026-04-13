import argparse
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

# Allow local imports even when run from other directories
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from helpers import _load_training_config_module

era = ""

mp.set_sharing_strategy("file_system")

MODES = ["train", "train_submit", "plot", "infer_iter", "infer"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select working mode")
    parser.add_argument("--working_mode", choices=MODES, help="select working mode")
    parser.add_argument(
        "--sample_folder_loc",
        dest="sample_folder_loc",
        type=str,
        default="",
        help="target sample folder",
    )
    parser.add_argument(
        "--input_model",
        dest="input_model",
        type=str,
        default="",
        help="input model to infer",
    )
    parser.add_argument(
        "--model_out_path", dest="out_path", type=str, help="training model output path"
    )
    parser.add_argument(
        "--branch_name",
        dest="branch_name",
        type=str,
        help="inferring branch name",
        default="template_score",
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        type=str,
        default="tensorrt",
        help="inference backend: pytorch, onnx, onnx-int8, tensorrt, tensorrt-int8",
    )
    parser.add_argument(
        "--result_folder_name",
        dest="result_folder_name",
        type=str,
        help="name of RunResult folder",
    )
    parser.add_argument(
        "--floss_gamma",
        dest="floss_gamma",
        type=float,
        default=0.0,
        help="focal loss gamm",
    )
    parser.add_argument(
        "--input_root_file",
        dest="input_root_file",
        type=str,
        help="file to infer",
    )
    parser.add_argument(
        "--pretrained_model",
        dest="pretrained_model",
        type=str,
        default=None,
        help="path of pretrained model",
    )
    parser.add_argument(
        "--era",
        dest="era",
        type=str,
        default=None,
        help="era",
    )
    parser.add_argument(
        "--add_year_index",
        dest="add_year_index",
        type=int,
        default=0,
        help="add year index",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=str,
        default=None,
        help="path to checkpoint model",
    )
    parser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        default=0,
        help="k-fold cross validation fold index",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to TrainingConfig.py. If omitted, imports the installed TrainingConfig module.",
    )
    parser.add_argument(
        "--local_infer_iter",
        action="store_true",
        help="Run infer_iter locally without HTCondor submission.",
    )
    parser.add_argument(
        "--infer_workers",
        dest="infer_workers",
        type=int,
        default=1,
        help="Workers for local infer_iter (1 = sequential).",
    )
    return parser


def main():
    global era
    parser = build_parser()
    args = parser.parse_args()
    era = args.era

    if args.add_year_index and (
        args.working_mode != "train" and args.working_mode != "train_submit"
    ):
        print("add_year_index is only available for train mode. It will be ignored.")

    if args.working_mode == "train":
        from mva import train

        print("Training Mode")
        train(
            model_save_path=args.out_path,
            floss_gamma=args.floss_gamma,
            result_folder_name=args.result_folder_name,
            sample_folder_loc=args.sample_folder_loc,
            pretrained_model=args.pretrained_model,
            checkpoint=args.checkpoint,
            add_year_index=args.add_year_index,
            fold=args.fold,
            TC=_load_training_config_module(args.config),
            era=era or "",
        )

    elif args.working_mode == "plot":
        from mva import plot

        print("Plotting Mode")
        plot(args.input_model, args.checkpoint)

    elif args.working_mode == "infer_iter":
        from mva import infer_with_iter

        print("Inffering Mode (all file iteration)")
        infer_with_iter(
            branch_name=args.branch_name,
            input_folder=args.sample_folder_loc,
            input_model_path=args.input_model,
            result_folder_name=args.result_folder_name,
            backend=args.backend,
            era=era,
            local=args.local_infer_iter,
            num_workers=args.infer_workers,
        )

    elif args.working_mode == "infer":
        import ROOT
        from mva import make_score_friend_file_parallel

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        ROOT.DisableImplicitMT()
        print("infering and writing")
        make_score_friend_file_parallel(
            input_root_file=args.input_root_file,
            input_model_path=args.input_model,
            out_group_name=args.branch_name,
            tree_name="Template_Training_Tree" if args.era == "2024" else "Result_Tree",
            backend=args.backend,
        )
    else:
        print("Wrong working mode")


if __name__ == "__main__":
    main()
