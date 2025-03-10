import pdb
import math

import numpy as np
import torch
from torch import nn

from . import relu
from . import siren
from . import wire


layer_dict = {
    'relu': relu.ReLULayer,
    'siren': siren.SineLayer,
    'wire': wire.ComplexGaborLayer
}


class MLPWithSkips(nn.Module):
    """
    after_network_type == 'MLP'에서 
    스킵 연결을 적용하기 위한 전용 MLP 모듈
    """
    def __init__(
        self,
        hidden_features,
        out_features,
        hidden_layers,
        nonlin,           # 예: relu.ReLULayer
        skips=[4, 8],     # 예시로 4의 배수마다 스킵
        R=1
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.nonlin = nonlin
        self.skips = skips

        layers = nn.ModuleList()
        for i in range(self.hidden_layers):
            if i == 0:
                # 첫 레이어: (hidden_features -> hidden_features)
                layers.append(
                    self.nonlin(self.hidden_features, self.hidden_features)
                )
            else:
                # 스킵 레이어면 in_dim = hidden_features + hidden_features
                if i in self.skips:
                    in_dim = self.hidden_features + self.hidden_features
                    out_dim = (
                        self.hidden_features * R
                        if i == (self.hidden_layers - 1)
                        else self.hidden_features
                    )
                    layers.append(self.nonlin(in_dim, out_dim))
                else:
                    in_dim = self.hidden_features
                    out_dim = (
                        self.hidden_features * R
                        if i == (self.hidden_layers - 1)
                        else self.hidden_features
                    )
                    layers.append(self.nonlin(in_dim, out_dim))

        # 마지막에 Linear로 out_features를 예측
        self.final_linear = nn.Linear(self.hidden_features, self.out_features)

        self.layers = layers

    def forward(self, x):
        """
        x.shape: (..., hidden_features)
        """
        input_h = x
        h = x
        # hidden_layers 만큼 반복
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                # skip 연결
                h = torch.cat([h, input_h], dim=-1)
            h = layer(h)

        # 마지막 Linear
        h = self.final_linear(h)
        return h


class INR_R2L(nn.Module):
    def __init__(
        self, 
        feat_per_channel=[2,1,1],
        coord_hidden_layers=8, 
        coord_hidden_features=256, 
        after_hidden_features=256, 
        after_hidden_layers=2, 
        out_features=3, 
        split_input_nonlin1=relu.ReLULayer,
        split_input_nonlin2=relu.ReLULayer,
        after_nonlin=relu.ReLULayer,
        before_nonlin=relu.ReLULayer,
        R=1,
        after_network_type="resnet",
        skip_connection = 1,
    ):
        super().__init__()

        self.feat_per_channel = feat_per_channel
        self.split_input_nonlin1 = split_input_nonlin1
        self.split_input_nonlin2 = split_input_nonlin2
        self.after_nonlin = after_nonlin
        self.before_nonlin = before_nonlin

        self.after_network_type = after_network_type

        self.coord_hidden_features = coord_hidden_features
        self.coord_hidden_layers = coord_hidden_layers
        self.after_hidden_features = after_hidden_features
        self.after_hidden_layers = after_hidden_layers
        self.reduce_method = "linear"

        # 각 채널마다 입력 레이어 생성
        self.coord_input_layer = nn.ModuleList(
            [nn.Linear(feat, coord_hidden_features) for feat in self.feat_per_channel]
        )

        # -----------------------------------------
        # coord_net에 대한 skip 리스트
        if skip_connection:
            self.skips = [4, 8, 12, 16]  # 예: 4의 배수 레이어마다 스킵
        else:
            self.skips = []

        # coord_net (MLP) 구성
        self.coord_net = nn.ModuleList()
        for i in range(self.coord_hidden_layers):
            if i == 0:
                self.coord_net.append(
                    self.after_nonlin(coord_hidden_features, coord_hidden_features)
                )
            else:
                if (i in self.skips):
                    in_dim = coord_hidden_features + coord_hidden_features
                    out_dim = coord_hidden_features * R if i == (self.coord_hidden_layers - 1) else coord_hidden_features
                    self.coord_net.append(self.after_nonlin(in_dim, out_dim))
                else:
                    in_dim = coord_hidden_features
                    out_dim = coord_hidden_features * R if i == (self.coord_hidden_layers - 1) else coord_hidden_features
                    self.coord_net.append(self.after_nonlin(in_dim, out_dim))

        # -----------------------------------------
        # after_network 구성 (MLP에도 skip 연결 적용)
        self.after_network = None
       

        if self.after_network_type == "MLP":
            # 여기서 MLPWithSkips 클래스를 사용해 skip 적용 가능
            # 예: 2층 이상일 때 1층에 skip 적용 등을 커스텀
            self.after_network = MLPWithSkips(
                hidden_features=after_hidden_features,
                out_features=out_features,
                hidden_layers=after_hidden_layers,
                nonlin=self.after_nonlin,
                skips=[ 4, 8]  # 원하는 대로 설정 (예: 2의 배수, 4의 배수 등)
            )

        elif self.after_network_type == "None":
            self.after_network = nn.Linear(coord_hidden_features, out_features)

        self.middle_output_layer = nn.Linear(coord_hidden_features, out_features)
        self.fusion_operator = 'prod'


    def forward(self, coords):
        """
        coords: list of tensors [coord_channel_0, coord_channel_1, ...]
        """
        hs = [self.forward_coord(coord, i) for i, coord in enumerate(coords)]
        h  = self.forward_fusion(hs)
        return h.reshape(-1, h.shape[-1])


    def forward_coord(self, coord, channel_id):
        """
        coord_net (MLP)에 스킵을 적용한 부분
        """
        input_h = self.coord_input_layer[channel_id](coord)

        h = input_h
        for i in range(self.coord_hidden_layers):
            layer = self.coord_net[i]
            if i in self.skips:
                # skip 연결
                h = torch.cat([h, input_h], dim=-1)
            h = layer(h)

        return h

    def forward_fusion(self, hs):
        """
        채널별 feature를 곱(prod)으로 융합
        """
        h = hs[0]
        for hi in hs[1:]:
            h = h * hi

        h_sh = h.shape

        if (self.after_network_type != 'None') and (h_sh[-1] > self.after_hidden_features):
            # 만약 hidden dim이 after_hidden_features보다 크면, sum(-1)로 축소
            h = h.reshape(*h_sh[:-1], self.after_hidden_features, -1).sum(-1)
            h_sh = h.shape

        # SR, resnet 모드는 4D 텐서(B, C, H, W)로 변환
        if self.after_network_type == "SR" or self.after_network_type == "resnet":
            h = h.view(-1, self.after_hidden_features, h_sh[2], h_sh[3])
            h = self.after_network(h)
            h = h.permute(0, 2, 3, 1)  # (B, H, W, C)

        # MLP (with skip) 후처리
        elif self.after_network_type == "MLP":
            h = self.after_network(h)
            h = h.squeeze(0)

        elif self.after_network_type == "None":
            h = self.after_network(h)
            h = h.squeeze(0)

        return h
