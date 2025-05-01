from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import pandas as pd
import os
import json

class VBenchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_csv, required_keys=None, scaler=None, fit_scaler=False):
        self.root_dir = root_dir
        self.label_df = pd.read_csv(label_csv)
        self.required_keys = required_keys
        self.scaler = scaler or StandardScaler()
        self.fit_scaler = fit_scaler
        self.samples = []
        self._load_data()

    def _load_data(self):
        raw_X = []
        raw_y = []

        for _, row in self.label_df.iterrows():
            folder_path = f"MOS_{row['Folder2']}"
            filename = row["FileName"]
            avg_score = row["Avg. Score"]

            metrics_path = os.path.join(self.root_dir, folder_path, "merged_with_all_metrics.json")
            if not os.path.exists(metrics_path):
                continue

            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)

                if filename not in metrics_data:
                    continue

                metrics = metrics_data[filename]
                if self.required_keys and not all(k in metrics for k in self.required_keys):
                    continue

                values = []
                for k in self.required_keys:
                    v = metrics[k]
                    values.append(float(v))

                raw_X.append(values)
                raw_y.append(avg_score)
            except Exception as e:
                print(f"⚠️ 오류 발생 - {metrics_path} → {e}")

        X_np = np.array(raw_X)
        y_np = np.array(raw_y)

        if self.fit_scaler:
            self.scaler.fit(X_np)
        X_scaled = self.scaler.transform(X_np)

        self.samples = list(zip(torch.tensor(X_scaled, dtype=torch.float32),
                                torch.tensor(y_np, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
