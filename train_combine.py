import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
import json
import os
from model import AttentionModel
from dataset import VBenchDataset
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# --- Preprocessed pairwise dataset (for inference only) ---
class PreprocessedPairwiseDataset(Dataset):
    def __init__(self, feature_dict_path, pairs_path, scaler, required_keys):
        with open(feature_dict_path, "r", encoding="utf-8") as f:
            raw_feature_dict = json.load(f)

        self.feature_dict = {}
        for fname, feats in raw_feature_dict.items():
            values = []
            for k in required_keys:
                v = feats[k]
                if isinstance(v, bool):
                    v = 0.9 if v else 0.1
                values.append(float(v))
            self.feature_dict[fname] = values

        with open(pairs_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

        self.scaler = scaler
        self.required_keys = required_keys

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2, label = self.pairs[idx]
        x1 = torch.tensor(self.scaler.transform([self.feature_dict[f1]])[0], dtype=torch.float32)
        x2 = torch.tensor(self.scaler.transform([self.feature_dict[f2]])[0], dtype=torch.float32)
        return x1, x2, torch.tensor(label, dtype=torch.float32)

# --- Pseudo + True 학습 ---
def train_with_pseudo_and_true(model, pseudo_dataset, true_dataset, epochs=10, lr=1e-4, margin=5.0, era=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pseudo_X, pseudo_y = [], []

    loader = DataLoader(pseudo_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x1, x2, label in loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            s1 = model(x1).item()
            s2 = model(x2).item()
            if abs(s1 - s2) < 1.0 : # Threshold for uncertainty
                continue  # 불확실한 비교는 제외
            avg = (s1 + s2) * 0.5

            if label.item() == 1 and s1 < s2:
                pseudo_X.append(x1.squeeze(0).cpu())
                pseudo_y.append(avg + margin)
                pseudo_X.append(x2.squeeze(0).cpu())
                pseudo_y.append(avg - margin)
            elif label.item() == -1 and s2 < s1:
                pseudo_X.append(x2.squeeze(0).cpu())
                pseudo_y.append(avg + margin)
                pseudo_X.append(x1.squeeze(0).cpu())
                pseudo_y.append(avg - margin)

    print(f"🧪 Era {era+1:03d} - Pseudo labels 생성 완료: {len(pseudo_X)}개")

    # MOS ground-truth 데이터
    x_true, y_true = [], []
    for x, y in true_dataset:
        x_true.append(x)
        y_true.append(y)
    x_true = torch.stack(x_true)
    y_true = torch.tensor([y.item() for y in y_true])


    # Combine
    X = torch.cat([x_true, torch.stack(pseudo_X)], dim=0)
    Y = torch.cat([y_true, torch.tensor(pseudo_y)], dim=0)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Fine-tuning
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Era {era+1:03d} | Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), f"models/fine_tuned_model_era{era+1:03d}.pth")
    print(f"✅ Era {era+1:03d} 모델 저장 완료")

# --- 실행 ---
if __name__ == "__main__":
    required_keys = [
        "overall_consistency", "subject_consistency", "background_consistency",
        "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality",
        "imaging_quality", "temporal_style", "inception_score", "face_consistency_score",
        "sd_score", "flow_score", "warping_error", "blip_bleu", "clip_score",
        "clip_temp_score", "aesthetic_score", "technical_score"
    ]

    scaler = joblib.load("models/standard_scaler_NEW.pkl")

    pseudo_dataset = PreprocessedPairwiseDataset(
        feature_dict_path="Dataset/feature_dict.json",
        pairs_path="Dataset/pairs_annotation.json",
        scaler=scaler,
        required_keys=required_keys
    )

    true_dataset = VBenchDataset(
        root_dir="./Dataset/MOS",
        label_csv="./Dataset/MOS_annotation.csv",
        required_keys=required_keys,
        scaler=scaler,
        fit_scaler=False
    )

    for era in range(500):
        print(f"--- Era {era:03d} ---")
        model = AttentionModel(input_dim=len(required_keys), hidden_dim=63)

        weight_path = (
            f"models/fine_tuned_model_era{era:03d}.pth"
            if era > 0 else "models/final_model.pth"
        )
        model.load_state_dict(torch.load(weight_path))
        print(f"\n🚀 Era {era+1:03d} 시작: 모델 불러오기 → {weight_path}")

        train_with_pseudo_and_true(
            model, pseudo_dataset, true_dataset,
            epochs=10, lr=1e-4, margin=1.0, era=era
        )
