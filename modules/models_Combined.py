import torch
import torch.nn as nn

class CoordSRCombined(nn.Module):

    def __init__(self, coordx_model, denoiser_model, scaled_h, scaled_w, input_channels, batch_size = 2):
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
        
    def forward(self, coords, batch_size=None, measure_time=False):
        if batch_size is None:
            batch_size = self.batch_size
            
        # 시간 측정을 위한 CUDA 이벤트 초기화
        if measure_time:
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_coordx = torch.cuda.Event(enable_timing=True)
            end_coordx = torch.cuda.Event(enable_timing=True)
            start_reshape = torch.cuda.Event(enable_timing=True)
            end_reshape = torch.cuda.Event(enable_timing=True)
            start_denoiser = torch.cuda.Event(enable_timing=True)
            end_denoiser = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_total.record()
        
        # coordx 모델 실행 시간 측정
        if measure_time:
            start_coordx.record()
            
        with torch.no_grad():
            rgb_features = self.coordx(coords)
            
        if measure_time:
            end_coordx.record()
            torch.cuda.synchronize()
        
        # reshape 및 permute 연산 시간 측정
        if measure_time:
            start_reshape.record()
            
        rgb_features = rgb_features.reshape((batch_size, self.scaled_h, self.scaled_w, self.input_channels))
        rgb_features = rgb_features.permute(0, 3, 1, 2)
        
        if measure_time:
            end_reshape.record()
            torch.cuda.synchronize()
        
        # denoiser 모델 실행 시간 측정
        if measure_time:
            start_denoiser.record()
            
        output = self.denoiser(rgb_features)
        
        if measure_time:
            end_denoiser.record()
            end_total.record()
            torch.cuda.synchronize()
            
            # 시간 출력 (밀리초 단위)
            print(f"coordx 모델 실행 시간: {start_coordx.elapsed_time(end_coordx):.2f} ms")
            print(f"reshape 및 permute 연산 시간: {start_reshape.elapsed_time(end_reshape):.2f} ms")
            print(f"denoiser 모델 실행 시간: {start_denoiser.elapsed_time(end_denoiser):.2f} ms")
            print(f"전체 forward 실행 시간: {start_total.elapsed_time(end_total):.2f} ms")
        
        return output
    
    def load_state_dicts(self, coordx_state_dict, denoiser_state_dict=None):

        self.coordx.load_state_dict(coordx_state_dict)
        if denoiser_state_dict is not None:
            self.denoiser.load_state_dict(denoiser_state_dict)
