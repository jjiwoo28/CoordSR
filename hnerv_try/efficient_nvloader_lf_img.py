import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from torchvision.io import write_video, read_image
from model_all import PositionEncoding

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'].to(torch.float32), quant_t['scale'].to(torch.float32)
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t
    
class ImageRenderer:
    def __init__(self, ckt, decoder, pe=True, grid_size=9, device='cuda:0', lbase=2.0, levels=32):
        self.grid_size = grid_size
        self.pe = pe
        
        self.pe_embed = PositionEncoding(f'pe_{lbase}_{levels}')
        
        # quant_ckt = torch.load(ckt, map_location='cpu', weights_only=True)

        # dequant_ckt = {k:dequant_tensor(v).to(device) for k,v in quant_ckt['model'].items()}

        # self.decoder = torch.jit.load(decoder, map_location='cpu').to(device)
        # self.decoder.load_state_dict(dequant_ckt)
        # print(self.decoder)

        self.decoder = torch.jit.load(decoder, map_location='cpu').to(device)
        print(self.decoder)

    def get_norm_input(self, start_x, start_y, end_x, end_y):
        steps = 50
        norm_y = (torch.linspace(start_y, end_y, steps, device=device) / self.grid_size).unsqueeze(1)
        norm_x = (torch.linspace(start_x, end_x, steps, device=device) / self.grid_size).unsqueeze(1)

        return torch.cat([norm_y, norm_x], dim=1)

    def render(self, start_x, start_y, end_x, end_y, frames):
        pe_input = self.get_norm_input(start_x, start_y, end_x, end_y)
        print(pe_input)
        input = self.pe_embed(pe_input)

        if device.startswith('cuda'):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            print("CUDA 타이머 시작")

        if device.startswith('cuda'):
            start_event.record()
            
        output = self.decoder(input)

        if device.startswith('cuda'):
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"GPU 연산 시간: {elapsed_time:.2f} 밀리초")
        
        return output

    def calculate_psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, default='checkpoints/img_decoder.pth', help='path for video decoder',)
    parser.add_argument('--ckt', type=str, default='checkpoints/quant_vid.pth', help='path for video checkpoint',)
    parser.add_argument('--dump_dir', type=str, default='visualize/bunny_1.5M_E300', help='path for video checkpoint',)
    parser.add_argument('--frames', type=int, default=16, help='video frames for output',)
    parser.add_argument('--grid_size', type=int, default=9, help='grid size',)
    parser.add_argument('--pe', type=bool, default=False, help='use pe',)
    parser.add_argument('--lbase', type=float, default=2.0, help='base for position encoding',)
    parser.add_argument('--levels', type=int, default=32, help='levels for position encoding',)
    parser.add_argument('--eval_interpolation', type=bool, default=False, help='evaluate interpolation',)
    parser.add_argument('--data_dir', type=str, default='', help='data directory for interpolation evaluation',)
    args = parser.parse_args()
    
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)


    video_renderer = ImageRenderer(args.ckt, args.decoder, args.pe, args.grid_size, device, args.lbase, args.levels)

    img_out = video_renderer.render(8, 0, 8, 15, args.frames).cpu()

    out_vid = os.path.join(args.dump_dir, 'nvloader_out.mp4')
    write_video(out_vid, img_out.permute(0,2,3,1) * 255., args.frames/4, options={'crf':'10'})

    print(f'dumped video to {out_vid}')

    # interpolation 평가 추가
    if args.eval_interpolation:
        print("\n인터폴레이션 평가 시작...")
        results = video_renderer.eval_interpolation(args.data_dir)

if __name__ == '__main__':
    main()
