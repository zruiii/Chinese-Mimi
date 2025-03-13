import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    def __init__(self, seanet_dim: int, hubert_dim: int):
        super().__init__()
        self.proj = nn.Linear(seanet_dim, hubert_dim)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ 计算语义蒸馏损失
        x: RVQ第一层输出, shape: [B, seanet_dim, F_x]
        y: WavLM特征, shape: [B, F_y, hubert_dim]
        """
        _, _, f_x = x.shape
        x = x.transpose(1, 2)                               # [B, F_x, seanet_dim]
        x_proj = self.proj(x)                               # [B, F_x, hubert_dim]

        # 按照论文 使用K8S4进行池化
        y_t = y.transpose(1, 2)                             # [B, hubert_dim, F_y]
        
        pad_size = 2
        y_padded = F.pad(y_t, (pad_size, pad_size), mode='replicate')
        y_pooled = F.avg_pool1d(y_padded, kernel_size=8, stride=4)
        y_pooled = y_pooled.transpose(1, 2)

        assert y_pooled.shape == x_proj.shape, f"wav embed: {x.shape}; hubert embed: {y.shape}"
        
        # 计算余弦相似度
        # x_norm = F.normalize(x_proj, p=2, dim=-1)
        # y_norm = F.normalize(y_pooled, p=2, dim=-1)
        
        # cosine_sim = torch.sum(x_norm * y_norm, dim=-1)     # [B, 25]
        # cosine_dist = 1 - cosine_sim

        cosine_dist = -torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(x_proj, y_pooled)))
        
        # 计算平均损失
        loss = cosine_dist.mean()
        
        return loss

# 使用示例
if __name__ == "__main__":
    # 创建随机测试数据
    batch_size = 4
    x = torch.randn(batch_size, 512, 25)
    y = torch.randn(batch_size, 100, 1024)
    
    # 初始化损失函数
    criterion = DistillLoss(512, 1024)
    
    # 计算损失
    loss = criterion(x, y)
    print(f"Distillation loss: {loss.item():.4f}")