import torch
import torch.nn as nn
import time
import torch.nn.functional as F

def get_activation(act_name='relu'):
    """원하는 activation 이름을 입력받아 torch.nn 모듈을 반환하는 헬퍼 함수입니다."""
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    elif act_name == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError(f"Activation [{act_name}] is not implemented")

class ResnetBlock2(nn.Module):
    """
    Define a Resnet block (2 conv layers + skip connection).
    참고: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(
        self, 
        dim,               # 채널 수
        kernel_size=3,     # conv3x3를 주로 사용
        padding_type='reflect',
        norm_layer=nn.BatchNorm2d,
        use_bias=True,
        act=nn.ReLU(inplace=True)
    ):
        """Initialize the Resnet block."""
        super(ResnetBlock2, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, 
            kernel_size, 
            padding_type,
            norm_layer, 
            use_bias, 
            act
        )

    def build_conv_block(
        self, 
        dim, 
        kernel_size, 
        padding_type, 
        norm_layer, 
        use_bias, 
        act
    ):
        """2번의 Conv -> Norm -> Act 구조로 이루어진 블록 생성."""
        conv_block = []
        # --------- 첫 번째 Conv ---------
        p = kernel_size // 2  # e.g. 3x3 conv => padding=1
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(p)]
            padding = 0
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(p)]
            padding = 0
        elif padding_type == 'zero':
            padding = p
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                 padding=padding, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]
        if act is not None:
            conv_block += [act]

        # --------- 두 번째 Conv ---------
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(p)]
            padding = 0
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(p)]
            padding = 0
        elif padding_type == 'zero':
            padding = p
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                 padding=padding, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # Skip Connection: out = x + F(x)
        out = x + self.conv_block(x)
        return out



class ResnetBlock(nn.Module):
    """
        Define a Resnet block.
        Refer to "https://github.com/ermongroup/ncsn/blob/master/models/pix2pix.py"
    """

    def __init__(
        self, 
        dim,
        kernel_size=1, 
        padding_type='zero',
        norm_layer=nn.BatchNorm2d, 
        use_dropout=False, 
        use_bias=True, 
        act=None
    ):
        
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, kernel_size, padding_type,
            norm_layer, use_dropout, use_bias, act
        )

    def build_conv_block(
        self, 
        dim, 
        kernel_size, 
        padding_type, 
        norm_layer, 
        use_dropout, 
        use_bias, 
        act=nn.GELU()
    ):

        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(dim, momentum=0.1)]
        if act:
            conv_block += [act]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(dim, momentum=0.1)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections

        return out


class ResNetDenoiser(nn.Module):
    """
    간단한 ResNet 기반 영상 필터(노이즈 제거) 네트워크 예시
    입력: (B, 3, H, W), 출력: (B, 3, H, W)
    """
    def __init__(
        self, 
        input_channels=3,
        output_channels=3,
        base_channels=64,
        n_blocks=4,          # ResNet block 개수
        activation='relu'
    ):
        super(ResNetDenoiser, self).__init__()

        act_layer = get_activation(activation)

        # 1) 첫 Conv (downscale 없이 단순 feature mapping)
        #    -> [B, base_channels, H, W]
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, 
                      padding=3, bias=True),
            nn.BatchNorm2d(base_channels),
            act_layer
        )

        # 2) ResNet Blocks
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(
                ResnetBlock(
                    dim=base_channels,
                    kernel_size=3,
                    padding_type='reflect',
                    norm_layer=nn.BatchNorm2d,
                    use_bias=True,
                    act=act_layer
                )
            )
        self.resnet_blocks = nn.Sequential(*res_blocks)

        # 3) 출력 Conv
        #    -> [B, output_channels, H, W]
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=7, 
                      padding=3, bias=True),
            nn.Sigmoid()  # 최종 출력을 0~1로 맞춰줄 때 사용 (선택사항)
        )

    def forward(self, x):
        """x: (B,3,H,W)"""
        x = self.conv_in(x)
        x = self.resnet_blocks(x)
        x = self.conv_out(x)
        return x


class DnCNNDenoiser(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3, 
        base_channels=64,
        n_layers=17,        # DnCNN의 레이어 개수
        kernel_size=3,
        activation='relu'
    ):
        super(DnCNNDenoiser, self).__init__()
        
        act_layer = get_activation(activation)
        
        # padding 크기를 kernel_size에 맞게 자동 계산
        padding = kernel_size // 2
        
        # 1) 첫번째 Conv + ReLU
        layers = [
            nn.Conv2d(input_channels, base_channels, kernel_size=kernel_size, padding=padding, bias=True),
            act_layer
        ]
        
        # 2) (n_layers-2)개의 Conv + BN + ReLU 레이어
        for _ in range(n_layers):
            layers.extend([
                nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(base_channels),
                act_layer
            ])
            
        # 3) 마지막 Conv 레이어
        layers.append(
            nn.Conv2d(base_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=True)
        )
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x: (B,3,H,W) - 노이즈가 있는 입력 이미지
        return: (B,3,H,W) - 노이즈 성분 예측값 (residual learning)
        """
        noise = self.dncnn(x)
        return noise  # 잔차 학습: 원본 이미지에서 예측된 노이즈를 빼줌
    
    
class SuperResolution(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3, 
        base_channels=64,
        n_layers=17,        # DnCNN의 레이어 개수
        kernel_size=3,
        activation='relu'
    ):
        super(SuperResolution, self).__init__()
        
        act = get_activation('gelu')
        # ResNet 블록 타입 선택
        if kernel_size == 1:
            # kernel_size가 1인 경우 ResnetBlock 사용 (기본 구현)
            ResBlock = ResnetBlock
        else:
            # kernel_size가 3 등인 경우 ResnetBlock2 사용 (conv3x3 구현)
            ResBlock = ResnetBlock2
        n_conv = 2# 2 conv layers in each sr
        n_up_block = 3 # 3 sr blocls
        kernels = [64, 64, 16]

        up_blocks = []
        for i in range(n_up_block - 1):
            in_dim = input_channels if not i else kernels[i]
            up_blocks += [nn.ConvTranspose2d(in_dim, kernels[i], 4, stride=2, padding=1)]
            up_blocks += [ResBlock(kernels[i], act=act) for _ in range(n_conv)]

        # hard-coded up-sampling factors
        # 12x for colmap

        k, s, p = 4, 2, 1 # 4, 2, 1  -> x8
        #k, s, p = 3, 1, 1 # 3, 3, 0  -> x12
            
        up_blocks += [nn.ConvTranspose2d(kernels[1], kernels[-1], k, stride=s, padding=p)]
        up_blocks += [ResBlock(kernels[-1], act=act)  for _ in range(n_conv)]
        up_blocks += [nn.Conv2d(kernels[-1], output_channels, 1), nn.Sigmoid()]
        self.sr = nn.Sequential(*up_blocks )
        
        
    def forward(self, x):
        sr = self.sr(x)
        return sr
    
    
    
class SuperResolution_x4(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3, 
        base_channels=64,
        n_layers=17,        # DnCNN의 레이어 개수
        kernel_size=3,
        activation='relu'
    ):
        # 상위 클래스 호출 시 올바른 클래스명을 사용해야 함
        super(SuperResolution_x4, self).__init__()
        
        act = get_activation('gelu')
        # ResNet 블록 타입 선택: kernel_size에 따라 선택
        if kernel_size == 1:
            ResBlock = ResnetBlock
        else:
            ResBlock = ResnetBlock2
        
        n_conv = 2  # 각 업샘플 블록 내에서 2개의 convolution 레이어 사용
        
        # x4 업샘플링을 위해 총 2개의 업샘플링 블록 사용 (2^2 = 4)
        n_up_block = 2
        
        # 여기서는 두 블록만 사용하므로, kernels 리스트도 두 단계로 구성 (예시)
        kernels = [base_channels, base_channels // 2]
        
        up_blocks = []
        # 첫 번째 업샘플링 블록
        in_dim = input_channels
        up_blocks += [
            nn.ConvTranspose2d(in_dim, kernels[0], kernel_size=4, stride=2, padding=1)
        ]
        # 첫 번째 블록의 conv 레이어들
        up_blocks += [ResBlock(kernels[0], act=act) for _ in range(n_conv)]
        
        # 두 번째(마지막) 업샘플링 블록
        # 이 블록이 최종 x2 업샘플링으로 총 x4가 됨 (2x2)
        up_blocks += [
            nn.ConvTranspose2d(kernels[0], kernels[1], kernel_size=4, stride=2, padding=1)
        ]
        # 두 번째 블록의 conv 레이어들
        up_blocks += [ResBlock(kernels[1], act=act) for _ in range(n_conv)]
        
        # 마지막에 출력 채널 수로 변환하고, Sigmoid를 적용
        up_blocks += [nn.Conv2d(kernels[1], output_channels, kernel_size=1), nn.Sigmoid()]
        self.sr = nn.Sequential(*up_blocks )
        
    def forward(self, x):
        sr = self.sr(x)
        return sr

class SuperResolution_x2(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3, 
        base_channels=64,
        n_layers=17,   # n_layers 인자를 추가하여 인터페이스 일치
        kernel_size=3,
        activation='relu'
    ):
        super(SuperResolution_x2, self).__init__()
        
        act = get_activation('gelu')
        # kernel_size에 따라 사용할 ResNet 블록 선택
        if kernel_size == 1:
            ResBlock = ResnetBlock
        else:
            ResBlock = ResnetBlock2
            
        n_conv = 2  # 각 업샘플 블록 내에 적용할 conv 레이어 수
        
        up_blocks = []
        # 2배 업샘플링을 위한 단일 ConvTranspose2d 레이어
        up_blocks += [nn.ConvTranspose2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1)]
        # ConvTranspose2d 뒤에 ResBlock을 n_conv번 적용
        up_blocks += [ResBlock(base_channels, act=act) for _ in range(n_conv)]
        # 마지막에 출력 채널 수로 변환한 후 Sigmoid로 활성화
        up_blocks += [nn.Conv2d(base_channels, output_channels, kernel_size=1), nn.Sigmoid()]
        self.sr = nn.Sequential(*up_blocks)
        
    def forward(self, x):
        sr = self.sr(x)
        return sr


# 채널 주의력(Channel Attention) 모듈
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 채널 감소 비율 조정 모델
class SuperResolution_x4_ChannelReduction(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, 
                 reduction_factor=2, activation='relu'):
        super(SuperResolution_x4_ChannelReduction, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
            
        # 첫 번째 업샘플링
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            ResnetBlock2(base_channels, act=act),
            ResnetBlock2(base_channels, act=act),
            # 채널 감소
            nn.Conv2d(base_channels, base_channels // reduction_factor, kernel_size=1)
        )
        
        # 두 번째 업샘플링
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // reduction_factor, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return x

# 비대칭 ResBlock 구성 모델
class SuperResolution_x4_Asymmetric(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, 
                 blocks1=2, blocks2=1, activation='relu'):
        super(SuperResolution_x4_Asymmetric, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        # 첫 번째 업샘플링 (ResnetBlock2 사용)
        layers = [nn.ConvTranspose2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1)]
        for _ in range(blocks1):
            layers.append(ResnetBlock2(base_channels, act=act))
        self.up1 = nn.Sequential(*layers)
        
        # 두 번째 업샘플링 (ResnetBlock 사용)
        layers = [nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1)]
        for _ in range(blocks2):
            layers.append(ResnetBlock(base_channels // 2, act=act))  # 1x1 컨볼루션
        layers.append(nn.Conv2d(base_channels // 2, output_channels, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.up2 = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return x



# 하이브리드 블록 구성 모델
class SuperResolution_x4_Hybrid(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, activation='relu'):
        super(SuperResolution_x4_Hybrid, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        # 첫 번째 업샘플링: 3x3 컨볼루션 (품질 중심)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            ResnetBlock2(base_channels, act=act),  # 3x3 컨볼루션
            ResnetBlock2(base_channels, act=act),
            nn.Conv2d(base_channels, base_channels // 4, kernel_size=1)  # 채널 감소
        )
        
        # 두 번째 업샘플링: 1x1 컨볼루션 (속도 중심)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
            ResnetBlock(base_channels // 8, act=act),  # 1x1 컨볼루션
            nn.Conv2d(base_channels // 8, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return x

# 채널 주의력 메커니즘 모델
class SuperResolution_x4_Attention(nn.Module):
    def __init__(self, input_channels=256, output_channels=3, base_channels=64, activation='relu'):
        super(SuperResolution_x4_Attention, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        # 특징 추출
        self.feature_extract = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            ResnetBlock2(base_channels, act=act)
        )
        
        # 채널 주의력 메커니즘
        self.attention = ChannelAttention(base_channels)
        
        # 업샘플링 (x4 = 2^2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            ResnetBlock2(base_channels // 2, act=act),
            nn.ConvTranspose2d(base_channels // 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # 타겟 크기 (512x512)
        self.target_size = (512, 512)
        
    def forward(self, x):
        # 특징 추출
        feat = self.feature_extract(x)
        
        # 주의력 적용
        attention_weights = self.attention(feat)
        feat = feat * attention_weights
        
        # 업샘플링
        out = self.up(feat)
        
        # 타겟 크기에 맞게 조정
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
        
        return out 

class SuperResolution_x2_PixelShuffle(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, activation='relu'):
        super(SuperResolution_x2_PixelShuffle, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        # 특징 추출
        self.feature_extract = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            ResnetBlock2(base_channels, act=act),
            ResnetBlock2(base_channels, act=act)
        )
        
        # x2 업샘플링
        self.upscale = nn.Sequential(
            nn.Conv2d(base_channels, output_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        features = self.feature_extract(x)
        out = self.upscale(features)
        return torch.sigmoid(out)
    
# 픽셀 셔플 기반 업샘플링 모델
class SuperResolution_x4_PixelShuffle(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, activation='relu'):
        super(SuperResolution_x4_PixelShuffle, self).__init__()
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        # 특징 추출
        self.feature_extract = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            ResnetBlock2(base_channels, act=act),
            ResnetBlock2(base_channels, act=act)
        )
        
        # 첫 번째 x2 업샘플링
        self.upscale1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            act
        )
        
        # 중간 처리
        self.mid_process = ResnetBlock2(base_channels, act=act)
        
        # 두 번째 x2 업샘플링
        self.upscale2 = nn.Sequential(
            nn.Conv2d(base_channels, output_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        features = self.feature_extract(x)
        mid = self.upscale1(features)
        mid = self.mid_process(mid)
        out = self.upscale2(mid)
        return torch.sigmoid(out)


class SuperResolution_x8_PixelShuffle(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, activation='relu'):
        super(SuperResolution_x8_PixelShuffle, self).__init__()
        
        # 활성화 함수 설정
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        # 특징 추출
        self.feature_extract = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            ResnetBlock2(base_channels, act=act),
            ResnetBlock2(base_channels, act=act)
        )
        
        # 첫 번째 x2 업샘플링
        self.upscale1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            act
        )
        
        # 첫 번째 중간 처리
        self.mid_process1 = ResnetBlock2(base_channels, act=act)
        
        # 두 번째 x2 업샘플링
        self.upscale2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            act
        )
        
        # 두 번째 중간 처리
        self.mid_process2 = ResnetBlock2(base_channels, act=act)
        
        # 세 번째 x2 업샘플링
        self.upscale3 = nn.Sequential(
            nn.Conv2d(base_channels, output_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        features = self.feature_extract(x)
        
        # 첫 번째 업스케일링 (x2)
        mid1 = self.upscale1(features)
        mid1 = self.mid_process1(mid1)
        
        # 두 번째 업스케일링 (x4)
        mid2 = self.upscale2(mid1)
        mid2 = self.mid_process2(mid2)
        
        # 세 번째 업스케일링 (x8)
        out = self.upscale3(mid2)
        
        return torch.sigmoid(out)

