import torch

class AttentionBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.query = torch.nn.Linear(input_dim, hidden_dim)
        self.key = torch.nn.Linear(input_dim, hidden_dim)
        self.value = torch.nn.Linear(input_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)


        attn_scores = torch.matmul(Q.transpose(-2,-1), K) / (Q.shape[-1] ** 0.5)  # Scaled Dot-Product Attention
        attn_weights = self.softmax(attn_scores)
        out = torch.matmul(V, attn_weights)

        return out


class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim=19, hidden_dim=63):
        super(AttentionModel, self).__init__()
        self.attention = AttentionBlock(input_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.act = torch.nn.GELU()  # 더 부드러운 비선형성
        self.out = torch.nn.Linear(hidden_dim, 1)  # 최종 회귀 출력

    def forward(self, x):
        batch_size, dim = x.shape
        x_diag = torch.stack([torch.diag(sample) for sample in x])  # (B, D, D)

        x = self.attention(x_diag)              # (B, D, H)
        x = x.mean(dim=1)                       # Attention 출력 평균 Pooling (B, H)
        x = self.fc1(x)
        x = self.act(x)
        x = self.out(x)                         # (B, 1)
        return x.squeeze(-1)                    # (B,)
    

def analyze_model_parameters(model):
    total_params = 0
    trainable_params = 0

    print(f"{'Layer':40s} {'# Params':>12s} {'Trainable':>10s}")
    print("-" * 65)

    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        total_params += num_params
        if is_trainable:
            trainable_params += num_params

        train_status = "Yes" if is_trainable else "No"
        print(f"{name:40s} {num_params:12,d} {train_status:>10s}")

    print("-" * 65)
    print(f"📦 총 파라미터 수: {total_params:,}")
    print(f"🧠 학습 가능한 파라미터 수: {trainable_params:,}")


