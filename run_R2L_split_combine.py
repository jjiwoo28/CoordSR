#!/usr/bin/env python

import os
import sys
from tqdm import tqdm
import importlib
import time
import glob

import argparse

import numpy as np
from scipy import io
from PIL import Image

import matplotlib.pyplot as plt
#plt.gray()

import cv2
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn
import torch.nn.functional as F  # F 모듈 추가
import torchvision
from torch.optim.lr_scheduler import LambdaLR


from modules.models_CoordX import COORDX
from modules.models_R2L import R2L_body


from modules.models_Denoiser import ResNetDenoiser
from modules.models_Denoiser import DnCNNDenoiser

from modules.models_Denoiser import SuperResolution_x2
from modules.models_Denoiser import SuperResolution_x4

from modules.models_Denoiser import SuperResolution

from modules.models_Denoiser import SuperResolution_x2_PixelShuffle
from modules.models_Denoiser import SuperResolution_x4_PixelShuffle
from modules.models_Denoiser import SuperResolution_x8_PixelShuffle

from my_utils.logger import PSNRLogger

from  my_utils.decomposer import get_mgrid


from my_utils.decomposer import (
    generate_fixed_coords, 
    get_limited_files, 
    combine_coordinates,
    PseudoImageDataset, 
    PseudoImageDataset4D,
    generate_fixed_coords_fully
)



from torch.utils.data import Dataset, DataLoader

from modules.models_Combined import CoordSRCombined
from modules.models_Combined import R2LCombined


def parse_argument():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=2 )
    parser.add_argument('--width', type=int, default=256 )
    
    parser.add_argument('--coord_depth', type=int, default=2 )
    parser.add_argument('--coord_width', type=int, default=256 )
    parser.add_argument('--R', type=int, default=1 )
    
    
    parser.add_argument('--whole_epoch', type=int, default=100 )

    parser.add_argument('--data_dir', type=str , default="data/stanford_half/beans")
    parser.add_argument('--exp_dir', type=str , default="result/stanford_half/beans")
    
    parser.add_argument('--test_freq', type=int , default=10)
    parser.add_argument('--save_ckpt_path', type=int , default=1)
    parser.add_argument('--lr', type=float , default=5e-3)
    parser.add_argument('--batch_size',type=int, default = 4,help='normalize input')
    
    parser.add_argument('--save_test_img', action='store_true')
    parser.add_argument('--wire_tunable', action='store_true')
    parser.add_argument('--real_gabor', action='store_true')
    parser.add_argument('--benchmark', action='store_true' , default=True)
    parser.add_argument('--test_img_save_freq', type=int , default=-1)
    
    parser.add_argument('--lr_batch_preset', action='store_true')
    parser.add_argument('--schdule_type', type=str , default="linear")
    
    parser.add_argument('--nonlin', type=str , default="relu")
    parser.add_argument('--decom_dim', type=str , default="us")
  
    parser.add_argument("--gpu", default="0", type=str, help="Comma-separated list of GPU(s) to use.")
    
    parser.add_argument('--render_only', action='store_true' , default=False)
    parser.add_argument('--loadcheckpoint', action='store_true' , default=False)
    parser.add_argument('--sample_type', type=str , default="all")
    parser.add_argument('--skip_connection',type=int , default=1)
    parser.add_argument('--res_depth',type=int , default=4)
    parser.add_argument('--res_width',type=int , default=64)
    
    parser.add_argument('--pseudo_data_path', type=str, default='/data/result/250104_us_only_decom_test_all/250104_us_only_decom_test_all_relu_d8_w256_cd2_cd256_R1_8192_decom_dim_us_lr0.0005_knights/pseudo_data', help='의사 데이터셋 경로')
    parser.add_argument('--coordx_model_path', type=str, default="/data/result/250116_us_only_decom_down_scale/250116_us_only_decom_down_scale_relu_d0_w128_cd8_cd256_R1_8192_decom_dim_us_lr0.0005_knights", help='입력 coordx 모델 체크포인트 경로')
    
    parser.add_argument('--cnn_type', type=str, default='dncnn', choices=['resnet', 'dncnn' ,'sr' ,'sr_pixel_shuffle'], help='CNN 모델 타입 선택 (resnet 또는 dncnn)')
    parser.add_argument('--after_network_type', type=str, default='rgb', choices=['rgb', 'feature'], help='CNN 입력 채널 수')
    parser.add_argument('--after_network', type=str, default='R2L_body', choices=['COORDX', 'R2L_body' , 'MLP'], help='SR 전 네트워크')
    parser.add_argument('--sr_scale',type=int , default=8)
    parser.add_argument('--data_set', type=str, default='knights', help='데이터셋 선택')
    opt = parser.parse_args()
    return opt 



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    nonlin = opt.nonlin            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = opt.whole_epoch               # Number of SGD iterations

    learning_rate = opt.lr
    maxpoints = opt.batch_size       
    
    print(f"ps : {opt.pseudo_data_path}")
    print(f"coordx : {opt.coordx_model_path}")
    #qbreakpoint()
    print(f"learning_rate : {learning_rate}")
    print(f"batch_size : {maxpoints}")

    # args
    norm_fac = 1
    st_norm_fac = 1
    rep = 1
    data_root = opt.data_dir
    save_dir  = opt.exp_dir
    
    logger_path = os.path.join(save_dir , 'log')
    test_path = os.path.join(save_dir , 'test')
    ckpt_path = os.path.join(save_dir, 'checkpoint')
    
    logger = PSNRLogger(logger_path , opt.exp_dir.split('/')[-1])
    logger.set_metadata("depth",opt.depth)
    logger.set_metadata("width",opt.width)
    logger.set_metadata("coord_depth",opt.coord_depth)
    logger.set_metadata("coord_width",opt.coord_width)
    dataset_name = opt.data_dir.split('/')[-1]
    logger.set_metadata("dataset_name",dataset_name)
    logger.set_metadata("model_info",nonlin)
    logger.set_metadata("lr",learning_rate)
    logger.set_metadata("batch_size",opt.batch_size)

    
    logger.set_metadata("decom_dim", opt.decom_dim)
    logger.set_metadata("R", opt.R)
    logger.set_metadata("schdule_type", opt.schdule_type)

    logger.set_metadata("res_depth", opt.res_depth)
    logger.set_metadata("res_width", opt.res_width)
    logger.set_metadata("after_network_type", opt.after_network_type)
    logger.set_metadata("cnn_type", opt.cnn_type)
    logger.set_metadata("data_set", opt.data_set)
    logger.set_metadata("sr_scale", opt.sr_scale)
    logger.set_metadata("pseudo_data_path", opt.pseudo_data_path)
    logger.set_metadata("coordx_model_path", opt.coordx_model_path)
    logger.set_metadata("cnn_type", opt.cnn_type)
    logger.set_metadata("after_network", opt.after_network)
    
    
    logger.load_results()
    
    test_freq = opt.test_freq
    if opt.test_img_save_freq == -1:
        test_img_save_freq = test_freq
    else:
        test_img_save_freq = opt.test_img_save_freq
    paths =[]
    
    paths.append(logger_path)
    paths.append(test_path)
    paths.append(ckpt_path)
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)



    # load nelf data
    print(f"Start loading...")
    split='train'
    # center
    #breakpoint()
    uvst_whole = np.load(f"{data_root}/uvst{split}.npy") / norm_fac
    uvst_whole[:,2:] /= st_norm_fac

    # norm to 0 to 1
    uvst_min = uvst_whole.min()
    uvst_max = uvst_whole.max()
    uvst_whole = (uvst_whole - uvst_min) / (uvst_max - uvst_min) * 2 - 1.0

    image_path = os.path.join(data_root, 'images')

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

    # 첫 번째 이미지 파일 읽기
    if image_files:
        first_image_path = os.path.join(image_path, image_files[0])
        with Image.open(first_image_path) as img:
            w, h = img.size
            img_w,img_h = img.size

        print(f"Width: {w}, Height: {h}")
    else:
        print("No image files found in the directory.")

    number_of_val_imges = 9
    number_of_train_images = len(image_files) - number_of_val_imges

        
    #breakpoint()
    # center color
    scale = opt.sr_scale
    
    scale_factor = scale
    
    def downsample_image(image, scale=8):
        """
        이미지를 지정된 스케일로 다운샘플링합니다.
        image: (H, W, C) 또는 (N, H, W, C) 형태의 numpy 배열
        """
        if len(image.shape) == 4:  # (N, H, W, C)
            N, H, W, C = image.shape
            new_H, new_W = H // scale, W // scale
            downsampled = np.zeros((N, new_H, new_W, C))
            for i in range(N):
                downsampled[i] = cv2.resize(image[i], (new_W, new_H), interpolation=cv2.INTER_AREA)
            return downsampled
        else:  # (H, W, C)
            H, W, C = image.shape
            return cv2.resize(image, (W // scale, H // scale), interpolation=cv2.INTER_AREA)

    
    color_whole  = np.load(f"{data_root}/rgb{split}.npy")

  

    #uvst_whole  = np.concatenate([uvst_whole]*rep, axis=0)
    color_whole = np.concatenate([color_whole]*rep, axis=0)

    split='val'
    uvst_whole_val  = np.load(f"{data_root}/uvst{split}.npy") / norm_fac
    uvst_whole_val[:,2:] /= st_norm_fac
    color_whole_val = np.load(f"{data_root}/rgb{split}.npy")
    uvst_whole_val  = (uvst_whole_val - uvst_min) / (uvst_max - uvst_min) * 2 - 1.0


    

    uvst_whole_val  = np.concatenate([uvst_whole_val]*rep, axis=0)
    color_whole_val = np.concatenate([color_whole_val]*rep, axis=0)
    print("Stop loading...")

    if color_whole.size == number_of_train_images * h * w * 3:
        color_whole = color_whole.reshape((number_of_train_images, h, w, 3))
        print("Reshaped array:", color_whole.shape)
    else:
        print("요소 수가 맞지 않아 배열을 해당 형태로 변환할 수 없습니다.")

    #breakpoint()

    if color_whole_val.size == number_of_val_imges * h * w * 3:
        color_whole_val = color_whole_val.reshape((number_of_val_imges, h, w, 3))
        print("Reshaped array:", color_whole_val.shape)
    else:
        print("요소 수가 맞지 않아 배열을 해당 형태로 변환할 수 없습니다.")

    image_axis_num = 17

    img_whole_num =289
    temp_whole = []
    val_idx_table = {
        72: 0, 76: 1, 80: 2,
        140: 3, 144: 4, 148: 5,
        208: 6, 212: 7, 216: 8
    }
    val_idx = [72, 76, 80, 140, 144, 148, 208, 212, 216]  # 기존 인덱스 유지
    j = 0
    i = 0
    for idx in range(img_whole_num):
        if idx in val_idx:
            temp_whole.append(color_whole_val[i,:,:,:])
            i+=1
        else:
            temp_whole.append(color_whole[j,:,:,:])
            j+=1

    color_whole = np.array(temp_whole)
    color_whole_original = color_whole
    
    
    color_whole = downsample_image(color_whole, scale=scale)
    w, h = img_w // scale, img_h // scale
    scaled_w, scaled_h = img_w // scale, img_h // scale
    #breakpoint()


    

    #color_whole = np.concatenate([color_whole , color_whole_val], axis=0)
    #breakpoint()
    color_whole = color_whole.reshape((image_axis_num,image_axis_num, h, w, 3))
    
    if opt.decom_dim == "uv":
        color_whole = color_whole.transpose(0, 1, 2, 3, 4)
    elif opt.decom_dim == "us":
        color_whole = color_whole.transpose(0, 2, 1, 3, 4)
        


    team1 = [0,1]
    team2 = [2,3]

    team1_shape = 1
    team2_shape = 1
    for s in team1:
        team1_shape*=color_whole.shape[s]

    for s in team2:
        team2_shape*=color_whole.shape[s]

    mgrid_team1 = get_mgrid((color_whole.shape[0], color_whole.shape[1]), dim=2).unsqueeze(1).cuda()
    mgrid_team2 = get_mgrid((color_whole.shape[2], color_whole.shape[3]), dim=2).unsqueeze(0).cuda()

    N_samples = opt.batch_size
    N_samples_dim = np.sqrt(N_samples)

    # prefix = 1
    # train_val_index = [4,8,12]
    # for i in train_val_index:
    #     for j in train_val_index:
    #         img_arr = (color_whole[i,j,:,:,:]*255).astype(np.uint8)

    #         img = Image.fromarray(img_arr)
    #         img.save(f'image_train_val_{prefix}.png')
    #         prefix+=1
            

    color_whole = color_whole.reshape((image_axis_num*image_axis_num*h*w, -1))
    color_whole = torch.tensor(color_whole).cuda()
    color_whole_val = torch.tensor(color_whole_val).cuda()

    def sampling_val_uv(xy_idx ,device='cuda'):
        coord_id1 = np.array([xy_idx]) 
        coord_id2 = np.arange(team2_shape)

        coord_id = (coord_id1[:,None] * team2_shape + coord_id2[None, :]).reshape(-1)
        
        # color_whole가 이미 GPU에 있으므로 .to(device)를 추가할 필요가 없음
        sampled_data = color_whole[coord_id,:]
        #sampled_data = sampled_data.reshape((h,w,-1))

        coord_1 = mgrid_team1[None , coord_id1 , : ,:]
        coord_2 = mgrid_team2[None , : , : ,:]

        return [coord_1 ,coord_2], sampled_data

    def sampling_val_us(xy_idx ,device='cuda'):
        x_idx = xy_idx % 17
        y_idx = xy_idx // 17
        coord_id1 = np.arange(h) + x_idx*h
        coord_id2 = np.arange(w) + y_idx*w
        
        
        # coord_id1 = np.array([xy_idx]) 
        # coord_id2 = np.arange(team2_shape)

        coord_id = (coord_id1[:,None] * team2_shape + coord_id2[None, :]).reshape(-1)
        
        # color_whole가 이미 GPU에 있으므로 .to(device)를 추가할 필요가 없음
        sampled_data = color_whole[coord_id,:]
        #sampled_data = sampled_data.reshape((h,w,-1))

        coord_1 = mgrid_team1[None , coord_id1 , : ,:]
        coord_2 = mgrid_team2[None , : , coord_id2 ,:]

        return [coord_1 ,coord_2], sampled_data
    
    location_1 = np.linspace(-1, 1, image_axis_num, dtype=np.float32)
    location_2 = np.linspace(-1, 1, image_axis_num, dtype=np.float32)
    
    
    def sampling_val_us_SR(xy_idx ,device='cpu'):
        x_idx = xy_idx % 17
        y_idx = xy_idx // 17
        coord_id1 = np.arange(h) + x_idx*h
        coord_id2 = np.arange(w) + y_idx*w
        
        local_coord_1 = location_1[x_idx]
        local_coord_2 = location_2[y_idx]
        
        coord_1, coord_2 = generate_fixed_coords(local_coord_2, local_coord_1, img_w, img_h, scale, device)
        val_idx = val_idx_table[xy_idx]
        
        sampled_data = color_whole_val[val_idx,:]
        sampled_data = sampled_data.reshape((-1,3))  # N,3 차원으로 변환
        
        
        return [coord_1 ,coord_2], sampled_data
    
    
    
    def generate_fixed_coords(x_fixed, y_fixed, img_w, img_h, scale, device='cpu'):

    
        coord_1 = torch.tensor([[[[x_fixed, y] for y in torch.linspace(-1, 1, img_h//scale)]]], device=device).permute(0, 2, 1, 3)

        # coord_2 생성: y_fixed를 고정하고 x 값을 -1부터 1까지 img_w 갯수만큼 생성
        coord_2 = torch.tensor([[[[ y_fixed,x] for x in torch.linspace(-1, 1, img_w//scale)]]], device=device)
        return coord_1, coord_2
    
    

    def generate_random_coords(num_samples, img_w, img_h, device='cuda'):
        random_coords = []
        for _ in range(num_samples):
            # -1부터 1 사이의 랜덤한 x_fixed와 y_fixed 값을 생성
            x_fixed = torch.rand(1).item() * 2 - 1  # [-1, 1] 범위로 변환
            y_fixed = torch.rand(1).item() * 2 - 1  # [-1, 1] 범위로 변환
            
            # generate_fixed_coords 함수를 사용하여 좌표 생성
            coord_1, coord_2 = generate_fixed_coords(x_fixed, y_fixed, img_w, img_h, device)
            random_coords.append((coord_1, coord_2))
        
        return random_coords
    
    def sampling_val_fully(xy_idx ,device='cpu'):
        x_idx = xy_idx % 17
        y_idx = xy_idx // 17
        #breakpoint()
        
        local_coord_1 = location_1[x_idx]
        local_coord_2 = location_2[y_idx]
        
        coord = generate_fixed_coords_fully(local_coord_2, local_coord_1, img_w, img_h, scale, device)
        coord = coord.unsqueeze(0)
        val_idx = val_idx_table[xy_idx]
        
        sampled_data = color_whole_val[val_idx,:]
        sampled_data = sampled_data.reshape((-1,3))  # N,3 차원으로 변환
        return coord, sampled_data
    #breakpoint()


    
    sampling_val = sampling_val_fully
    #breakpoint()
    cs ,  img_val = sampling_val(72)
    for i in range(len(cs)):
        print(f"cs[{i}].shape : {cs[i].shape}")
        
    #breakpoint()
    if opt.after_network == 'COORDX':
        model = COORDX(feat_per_channel=[2,2],
                    coord_hidden_features = opt.coord_width,
                    coord_hidden_layers= opt.coord_depth, 
                    after_hidden_features= opt.width, 
                    after_hidden_layers = opt.depth, 
                    out_features=3, 
                    after_network_type = "None",

                    R = opt.R,
                    skip_connection = opt.skip_connection,
                    isTeacher = True,
                    )
    elif opt.after_network == 'R2L_body':
        model = R2L_body(isTeacher = True )
        
        
    input_ckpt_path = os.path.join(opt.coordx_model_path, "checkpoint")
    
    input_ckpt_paths = glob.glob(os.path.join(input_ckpt_path,"*.pth"))
    load_epoch = 0

    #breakpoint()
    
    if opt.loadcheckpoint or opt.render_only:
        if len(input_ckpt_paths) > 0:
            for path in input_ckpt_paths:
                print(input_ckpt_path)
                ckpt_id = int(os.path.basename(path).split("ep")[1].split(".")[0])
                load_epoch = max(load_epoch, ckpt_id)
            ckpt_name = f"{input_ckpt_path}/ep{load_epoch}.pth"
            # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
            print(f"Load weights from {ckpt_name}")
            ckpt = torch.load(ckpt_name)
                
            #breakpoint()
            model.load_state_dict(ckpt)
        else:
            assert False, "No checkpoint found"
    print(model)
    # Send model to CUDA
    model.cuda()
    


    
    
    
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    #     def __init__(
    #     self, 
    #     input_channels=3,
    #     output_channels=3,
    #     base_channels=64,
    #     n_blocks=4,          # ResNet block 개수
    #     activation='relu'
    # ):
    
    if opt.after_network_type == 'rgb':
        cnn_input_channels = 3
    elif opt.after_network_type == 'feature':
        cnn_input_channels = 256
    
    
    
    # --- (새로 추가할) ResNet Denoiser 초기화 ---
    if opt.cnn_type == 'resnet':
        denoiser = ResNetDenoiser(
            input_channels=cnn_input_channels,  # 정의부와 동일한 이름
            output_channels=3,
            base_channels=opt.res_width,
            n_blocks=opt.res_depth
        ).cuda()
    
    elif opt.cnn_type == 'dncnn':
        denoiser = DnCNNDenoiser(
            input_channels=cnn_input_channels,
            output_channels=3,
            base_channels=opt.res_width,
            n_layers=opt.res_depth,
            kernel_size=3
        ).cuda()
        
    elif opt.cnn_type == 'sr':
        if opt.sr_scale == 8:
            denoiser = SuperResolution(
                input_channels=cnn_input_channels,
                output_channels=3,
                base_channels=opt.res_width,
                n_layers=opt.res_depth,
                kernel_size=3
            ).cuda()
        elif opt.sr_scale == 4:
            denoiser = SuperResolution_x4(
                input_channels=cnn_input_channels,
                output_channels=3,
                base_channels=opt.res_width,
                n_layers=opt.res_depth,
                kernel_size=3
            ).cuda()
        elif opt.sr_scale == 2:
            denoiser = SuperResolution_x2(
                input_channels=cnn_input_channels,
                output_channels=3,
                base_channels=opt.res_width,
                 n_layers=opt.res_depth,
                kernel_size=3
            ).cuda()
    elif opt.cnn_type == 'sr_pixel_shuffle':
        if opt.sr_scale == 8:
            denoiser = SuperResolution_x8_PixelShuffle(
                input_channels=cnn_input_channels,
                output_channels=3,
                base_channels=opt.res_width,
                activation='relu'
            ).cuda()
        elif opt.sr_scale == 4:
            denoiser = SuperResolution_x4_PixelShuffle(
                input_channels=cnn_input_channels,
                output_channels=3,
                base_channels=opt.res_width,
                activation='relu'
            ).cuda()
        elif opt.sr_scale == 2:
            denoiser = SuperResolution_x2_PixelShuffle(
                input_channels=cnn_input_channels,
                output_channels=3,
                base_channels=opt.res_width,
                activation='relu'
            ).cuda()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    
    print(denoiser)
        #     input_channels=3,
        # output_channels=3,
        # base_channels=32,
        # n_blocks=4
  
    if opt.after_network == 'COORDX':
    # CoordX와 Denoiser를 결합한 새로운 모델 생성
        combined_model = CoordSRCombined(
            coordx_model=model, 
            denoiser_model=denoiser,
            scaled_h=scaled_h, 
            scaled_w=scaled_w, 
            input_channels=cnn_input_channels,
            batch_size=opt.batch_size
        ).cuda()
    elif opt.after_network == 'R2L_body':
        combined_model = R2LCombined(
            r2l_body_model=model, 
            denoiser_model=denoiser,
            scaled_h=scaled_h, 
            scaled_w=scaled_w, 
            input_channels=cnn_input_channels,
            batch_size=opt.batch_size
        ).cuda()

    
    # 옵티마이저는 이제 결합 모델의 파라미터(실질적으로는 denoiser의 파라미터)를 최적화
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=5e-4)
    
    num_epochs = opt.whole_epoch # 예시로 5 epoch
# Scheduler: CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )
    criterion = torch.nn.MSELoss()

    folder_path = opt.pseudo_data_path
    batch_size = opt.batch_size
    
    if opt.after_network == 'COORDX':
        pseudo_dataset = PseudoImageDataset(folder_path=folder_path , limit=1000, scale=scale)
        
    elif opt.after_network == 'R2L_body' or opt.after_network == 'MLP':
        pseudo_dataset = PseudoImageDataset4D(folder_path=folder_path , limit=1000, scale=scale)
    else:
        assert False, "Invalid after_nrgb_features = rgb_features.reshape((batch_size, self.scaled_h, self.scaled_w, self.input_channels))etwork"
    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    

    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        combined_model.denoiser.train()  # denoiser 부분만 훈련 모드로 설정
        running_loss = 0.0
        
        # tqdm을 사용하여 진행 상황 표시
        for idx, (coords, image) in enumerate(tqdm(pseudo_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            coords = coords.cuda()
            image = image.cuda()                # Ground Truth (원본)
            #breakpoint()
            # 올바른 코드
            #breakpoint()
            denoised = combined_model(coords, batch_size)  # 결합 모델을 통해 전체 파이프라인 실행
            
            # 3) Loss = MSE(denoised, GT)
            # 크기 불일치 확인 및 처리
            if denoised.shape != image.shape:
                # 출력 크기에 맞게 GT 이미지 크기 조정
                image = F.interpolate(image, size=(denoised.shape[2], denoised.shape[3]), mode='bilinear', align_corners=False)
                
            loss = criterion(denoised, image)
            
            # 4) Backprop (ResNet만 학습)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if idx % 300 == 0 and idx != 0:
            #     # PSNR 계산
            #     psnr_value = peak_signal_noise_ratio(image.cpu().numpy(), denoised.detach().cpu().numpy(), data_range=1)
            #     print(f"[Epoch {epoch+1}, Step {idx}/{len(pseudo_loader)}] loss: {loss.item():.4f}, PSNR: {psnr_value:.4f}")

                
        avg_loss = running_loss / len(pseudo_loader)
        scheduler.step()
        
        #print(f"==> Epoch {epoch+1}/{num_epochs}, Avg loss: {avg_loss:.4f}")
        
        if (epoch % opt.save_ckpt_path == 0) and (epoch != 0):
                #breakpoint()
                cpt_path = ckpt_path + f"/ep{epoch}.pth"
                torch.save(combined_model.denoiser.state_dict(), cpt_path)
                print(f"save ckpt {epoch} : {cpt_path}")    
        #if epoch % test_freq ==0:
        if epoch % opt.test_freq ==0 and epoch != 0:
                avg_inference_time = 0
                avg_backward_time = 0
                
                print(f"==> Epoch {epoch+1}/{num_epochs}, Avg loss: {avg_loss:.4f}")
                #print(f"epoch : {epoch}")
                i = 0
                count = 0
                psnr_arr = []
                val_idx = [72, 76, 80, 140, 144, 148, 208, 212, 216]
                #breakpoint()
                val_size = len(val_idx)
                
                # 출력 디렉토리 생성 (이 부분 추가)
                os.makedirs('output_result', exist_ok=True)
                
                for idx in val_idx:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    #breakpoint()
                    
                      # pseudo_loader에서 미니배치 하나 샘플
                    coords, image = sampling_val(idx)
                    coords = coords.cuda()
                    image = image

                    
                    
                    torch.cuda.synchronize()
                    start.record()
                    #breakpoint()
                    denoised = combined_model(coords, 1, measure_time=False)
                    end.record()
                    torch.cuda.synchronize()
                    
                    
                    avg_inference_time += (start.elapsed_time(end) / val_size)
                    # 첫 번째 이미지를 예시로 저장
                    image = image.reshape((img_h, img_w, 3)).permute((2, 0, 1))
                    
                    # 크기 불일치 확인 및 처리
                    if denoised.shape[2:] != image.shape[1:]:
                        # image를 denoised 크기에 맞게 조정
                        image = F.interpolate(image.unsqueeze(0), size=denoised.shape[2:], mode='bilinear', align_corners=False).squeeze(0)
                    
                    idx_to_save = 0
                    val_test_path = f"{test_path}/ep{epoch}"
                    os.makedirs(val_test_path, exist_ok=True)
                    #torchvision.utils.save_image(rgb_noisy[0],
                                                #f'{test_path}/epoch{epoch+1}_noisy.png')
                    torchvision.utils.save_image(denoised[idx_to_save],
                                                f'{val_test_path}/{idx}_denoised.png')
                    # numpy array를 tensor로 변환
                    #breakpoint()
                    torchvision.utils.save_image(image,
                                                f'{val_test_path}/{idx}_gt.png')
                    
                    torchvision.utils.save_image(denoised[idx_to_save],
                                                f'output_result/epoch{epoch+1}_denoised.png')
                    torchvision.utils.save_image(image,
                                                f'output_result/epoch{epoch+1}_gt.png')
                        
                  


                    denoised = denoised.squeeze(0) 
                    #breakpoint()
                    # PSNR 계산 전 크기 확인
                    if denoised.shape != image.shape:
                        # 크기가 다르면 맞춤
                        image = F.interpolate(image.unsqueeze(0), size=(denoised.shape[1], denoised.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
                    
                    psnr = peak_signal_noise_ratio(image.cpu().numpy(), denoised.detach().cpu().numpy(), data_range=1)
        #                   ssim = structural_similarity(gt_color.reshape((img_h,img_w,3)), pred_color.reshape((img_h,img_w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
        #                   lsp  = self.lpips(pred_img.cpu(),gt_img)
                    psnr_arr.append(psnr)
                    #print(psnr)
        #                   s.append(ssim)
        #                   l.append(np.asscalar(lsp.numpy()))
                    #breakpoint()
                    i = end
                    count+=1
         
                
                    
                logger.push_infer_time(avg_inference_time ,epoch)    
                print(f"infer time : {avg_inference_time:.2f}")
                
                whole_psnr = 0
                for psnr in psnr_arr:
                    whole_psnr += psnr
                psnr_result = whole_psnr/count
                logger.push(psnr_result , epoch)
                
                psnr_arr_rounded = [f"{psnr:.2f}" for psnr in psnr_arr]

                print(f"epoch : {epoch:.2f} , PSNR -> avg : {psnr_result}  all : {psnr_arr_rounded}")
                
                logger.save_results()






if __name__ =="__main__":
    opt = parse_argument()
    
    run(opt)
