# -*- coding: utf-8 -*-
import torch
import torch.nn as nn



class CoordSRCombined(nn.Module):
    def __init__(self, coordx_model, denoiser_model, scaled_h, scaled_w, input_channels, batch_size=2):
        super(CoordSRCombined, self).__init__()
        self.coordx = coordx_model
        self.denoiser = denoiser_model
        self.scaled_h = scaled_h
        self.scaled_w = scaled_w
        self.input_channels = input_channels
        self.batch_size = batch_size
        
        for param in self.coordx.parameters():
            param.requires_grad = False
        self.coordx.eval()
        
    def forward(self, coords, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        #breakpoint()
        with torch.no_grad():
            rgb_features = self.coordx(coords)
            
        rgb_features = rgb_features.reshape((batch_size, self.scaled_h, self.scaled_w, self.input_channels))
        rgb_features = rgb_features.permute(0, 3, 1, 2)
        
        output = self.denoiser(rgb_features)
        
        return output
    
    def load_state_dicts(self, coordx_state_dict, denoiser_state_dict=None):
        self.coordx.load_state_dict(coordx_state_dict)
        if denoiser_state_dict is not None:
            self.denoiser.load_state_dict(denoiser_state_dict)

class R2LCombined(nn.Module):
    def __init__(self, r2l_body_model, denoiser_model, scaled_h, scaled_w, input_channels, batch_size = 2):
        super(R2LCombined, self).__init__()
        self.r2l_body = r2l_body_model
        self.denoiser = denoiser_model
        self.scaled_h = scaled_h
        self.scaled_w = scaled_w
        self.input_channels = input_channels
        self.batch_size = batch_size
        
        for param in self.r2l_body.parameters():
            param.requires_grad = False
        self.r2l_body.eval()
        
    def forward(self, coords, batch_size=None, measure_time=False):
        if batch_size is None:
            batch_size = self.batch_size
        with torch.no_grad():
            #breakpoint()
            coords = coords.squeeze(1)
            # coords가 리스트인 경우 첫 번째 요소만 사용
            
            # 입력 데이터 형태 확인 및 변환
            if len(coords.shape) == 4:
                # [B, H, W, C] 형태인 경우 [B, C, H, W]로 변환
                if coords.shape[-1] == 4:  # 마지막 차원이 채널(4)인 경우
                    coords = coords.permute(0, 3, 1, 2)
            
            # R2L_body 모델에 입력
            rgb_features = self.r2l_body(coords)
            
        # 출력 형태 확인 및 변환
        
        output = self.denoiser(rgb_features)
       
        
        return output
    
    def load_state_dicts(self, r2l_body_state_dict, denoiser_state_dict=None):
        self.r2l_body.load_state_dict(r2l_body_state_dict)
        if denoiser_state_dict is not None:
            self.denoiser.load_state_dict(denoiser_state_dict)