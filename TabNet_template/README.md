# Tabnet Training for Vcb Analysis
## How to use
### Training
```
python MVA.py --working_mode train --model_out_path [YOUR_MODEL_OUT_PATH] --sample_folder_loc=[PATH_TO_SAMPLE_PATH] --result_folder_name [WHICH_FOLDER_TO_CHOOSE] --era [ERA] --add_year_index=[1 if you want to add year index in feature] --config [TRAINING_CONFIG] --fold [FOLD INDEX]
```
### Inference
```
 python MVA.py --working_mode infer_iter --sample_folder_loc [PATH_TO_SAMPLE_PATH]  --result_folder_name [WHICH_FOLDER_TO_CHOOSE] --input_model [PATH_TO_MODEL] --branch_name 7Class --era 2024 --local_infer_iter --infer_workers 4 --backend [TORCH or ONNX or TENSORRT]
```

