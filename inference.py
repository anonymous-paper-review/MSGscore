import torch
import json
import csv
from model import AttentionModel, analyze_model_parameters
import warnings
import joblib
import numpy as np

def load_metrics(json_path, required_keys, scaler=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inputs = []
    video_names = []
    raw_inputs = []

    for video_name, metrics in data.items():
        if all(k in metrics for k in required_keys):
            values = []
            for k in required_keys:
                v = metrics[k]
                values.append(float(v))
            inputs.append(values)
            raw_inputs.append(values)
            video_names.append(video_name)
        else:
            print(f"⚠️ 누락된 metric → {video_name}")

    inputs_np = np.array(inputs)
    if scaler:
        inputs_np = scaler.transform(inputs_np)

    return torch.tensor(inputs_np, dtype=torch.float32), video_names, raw_inputs

def run_inference(model_path, metrics_json_path, required_keys, output_csv_path, scaler_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionModel(input_dim=len(required_keys), hidden_dim=63).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = joblib.load(scaler_path) if scaler_path else None
    inputs, video_names, raw_inputs = load_metrics(metrics_json_path, required_keys, scaler)
    inputs = inputs.to(device)

    with torch.no_grad():
        preds = model(inputs)  # (B,)

    # CSV 저장
    with open(output_csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["filename"] + required_keys + ["prediction"]
        writer.writerow(header)

        for name, features, pred in zip(video_names, raw_inputs, preds):
            row = [name] + features + [round(pred.item(), 4)]
            writer.writerow(row)

    print(f"✅ 예측 결과가 CSV로 저장되었습니다 → {output_csv_path}")

# 사용 예시
if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

    required_keys = [
        "overall_consistency", "subject_consistency", "background_consistency",
        "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality",
        "imaging_quality", "temporal_style", "inception_score", "face_consistency_score", "sd_score",
        "flow_score", "warping_error", "blip_bleu", "clip_score",
        "clip_temp_score", "aesthetic_score", "technical_score"
    ]

    run_inference(
        model_path="./models/final_model.pth",
        metrics_json_path="./Human_made_videos_NoEdit.json",
        required_keys=required_keys,
        output_csv_path="./result/inference_results.csv",
        scaler_path="./models/standard_scaler_NEW.pkl"  # ✅ 학습 시 저장한 스케일러 경로
    )

    # analyze_model_parameters(model = AttentionModel(input_dim=len(required_keys), hidden_dim=63).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
