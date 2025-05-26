import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class NERVBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class NERVSuperResolution(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, dim=256, depth=4, heads=8, scale_factor=8):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 입력 특징 임베딩
        self.embedding = nn.Linear(in_channels, dim)
        
        # NERV 블록들
        self.blocks = nn.ModuleList([
            NERVBlock(dim=dim, heads=heads, mlp_dim=dim*4)
            for _ in range(depth)
        ])
        
        # 업스케일링 모듈 (픽셀 셔플 사용)
        self.upsample = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, out_channels * (scale_factor ** 2))
        )
        
    def forward(self, x):
        """
        입력: x with shape [B, H/8, W/8, C=256]
        출력: [B, H, W, 3]
        """
        B, H, W, C = x.shape
        
        # 특징 형태 변환 [B, H, W, C] -> [B, H*W, C]
        x = x.reshape(B, H*W, C)
        
        # 임베딩
        x = self.embedding(x)
        
        # NERV 블록 통과
        for block in self.blocks:
            x = block(x)
        
        # 업스케일링 준비
        x = self.upsample(x)
        
        # 형태 변환 및 픽셀 셔플 적용
        x = x.reshape(B, H, W, self.scale_factor, self.scale_factor, 3)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H*self.scale_factor, W*self.scale_factor, 3)
        
        return x

# 테스트 코드
def test_nerv_block():
    # 입력 데이터 생성 (h/8, w/8, 256)
    batch_size = 1
    h, w = 32, 32  # 다운샘플링된 해상도 (원본 해상도의 1/8)
    feature_dim = 256
    
    # 랜덤 특징 맵 생성
    x = torch.randn(batch_size, h, w, feature_dim)
    
    # NERV 슈퍼 해상도 모델 생성
    model = NERVSuperResolution(
        in_channels=feature_dim,
        out_channels=3,
        dim=feature_dim,
        depth=4,
        heads=8,
        scale_factor=8
    )
    
    # 추론
    with torch.no_grad():
        output = model(x)
    
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    
    # 시각화를 위한 랜덤 RGB 이미지 생성 (출력 형태와 동일)
    random_rgb = torch.rand_like(output)
    
    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(random_rgb[0].detach().cpu().numpy())
    plt.title("입력 저해상도 특징의 예상 출력")
    plt.subplot(1, 2, 2)
    plt.imshow(output[0].detach().cpu().numpy())
    plt.title("NERV 모델 출력")
    plt.savefig("nerv_output_comparison.png")
    plt.show()
    
    return x, output, model

# NERV 블록의 파라미터 수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 메인 테스트 실행
if __name__ == "__main__":
    x, output, model = test_nerv_block()
    print(f"모델 파라미터 수: {count_parameters(model):,}")
    
    # NERV와 기존 SR의 차이점 테스트
    # 간단한 SR 모델 생성 (비교용)
    simple_sr = nn.Sequential(
        nn.Linear(256, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * 64)  # 8x8=64 픽셀당 업스케일
    )
    
    # 파라미터 수 비교
    nerv_params = count_parameters(model)
    sr_params = count_parameters(simple_sr)
    
    print(f"NERV 모델 파라미터: {nerv_params:,}")
    print(f"간단한 SR 모델 파라미터: {sr_params:,}")
    print(f"NERV/SR 파라미터 비율: {nerv_params/sr_params:.2f}x")
