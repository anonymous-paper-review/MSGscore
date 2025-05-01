import torch
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from model import AttentionModel
from dataset import VBenchDataset  # 정규화 지원 버전 사용


# Training Function
def train_model(model, dataloader, epochs=500, lr=0.001, save_path="models"):
    os.makedirs(save_path, exist_ok=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % (epochs // 5) == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
            

    # Save final model
    final_model_path = os.path.join(save_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved at {final_model_path}")


# Main Execution
if __name__ == "__main__":
    required_features = [
        "overall_consistency", "subject_consistency", "background_consistency",
        "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality",
        "imaging_quality", "temporal_style", "inception_score", "face_consistency_score", "sd_score",
        "flow_score", "warping_error", "blip_bleu", "clip_score",
        "clip_temp_score", "aesthetic_score", "technical_score"
    ]

    scaler = StandardScaler()

    dataset = VBenchDataset(
        root_dir="./Dataset/MOS",
        label_csv="./Dataset/MOS_annotation.csv",
        required_keys=required_features,
        scaler=scaler,
        fit_scaler=True  # ✅ 학습 데이터로 정규화 fit
    )

    # Save the fitted scaler
    joblib.dump(scaler, "models/standard_scaler.pkl")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = AttentionModel()

    print(f"📦 MOS 샘플 수: {len(dataset)}")

    weight_path = "models/final_model.pth"
    if os.path.exists(weight_path):
        print(f"📦 Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path))

    train_model(model, loader, epochs=500, lr=0.005)
    
    print("✅ Training completed.")
