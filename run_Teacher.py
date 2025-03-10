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
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ssim

from modules import models_Teacher as models_decom
#from modules import utils


from my_utils.logger import PSNRLogger

from my_utils.decomposer import get_mgrid

import imageio
import torchvision.utils as vutils

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
    parser.add_argument('--batch_size',type=int, default = 256*256,help='normalize input')
    
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
    opt = parser.parse_args()
    return opt 



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    nonlin = opt.nonlin            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = opt.whole_epoch               # Number of SGD iterations
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
          # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = opt.depth      # Number of hidden layers in the MLP
    hidden_features = opt.width   # Number of hidden units per layer
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
#    im = utils.normalize(plt.imread('data/parrot.png').astype(np.float32), True)
#    im = cv2.resize(im, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
#    H, W, _ = im.shape
    
    # Create a noisy image
#    im_noisy = utils.measure(im, noise_snr, tau)
    if opt.lr_batch_preset:
        if opt.nonlin =="relu" or opt.nonlin =="relu_skip" or opt.nonlin =="relu_skip2":
            learning_rate =0.0005
            maxpoints = 8192
            
        elif opt.nonlin =="wire": 
        
            learning_rate =0.005
            maxpoints = 65536
                
            
        elif opt.nonlin =="siren": 
            learning_rate =0.0005
            maxpoints = 8192
            
        elif opt.nonlin =="gauss": 
            #after에서는 0.005가 잘 작동함.
            learning_rate =0.0005
            maxpoints = 8192
            
        elif opt.nonlin =="finer": 
            learning_rate =0.0005
            maxpoints = 8192
            
            
            
            
    else:
        learning_rate = opt.lr
        maxpoints = opt.batch_size       
    
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
    color_whole  = np.load(f"{data_root}/rgb{split}.npy")
    trans        = np.load(f"{data_root}/trans{split}.npy")
    intrinsic    = np.load(f"{data_root}/k{split}.npy")
    fdepth       = np.load(f"{data_root}/fdepth{split}.npy") # center object
    render_pose  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
    st_depth     = -fdepth

    #uvst_whole  = np.concatenate([uvst_whole]*rep, axis=0)
    color_whole = np.concatenate([color_whole]*rep, axis=0)

    split='val'
    uvst_whole_val  = np.load(f"{data_root}/uvst{split}.npy") / norm_fac
    uvst_whole_val[:,2:] /= st_norm_fac
    color_whole_val = np.load(f"{data_root}/rgb{split}.npy")
    uvst_whole_val  = (uvst_whole_val - uvst_min) / (uvst_max - uvst_min) * 2 - 1.0

    trans_val        = np.load(f"{data_root}/trans{split}.npy")
    intrinsic_val    = np.load(f"{data_root}/k{split}.npy")
    fdepth_val       = np.load(f"{data_root}/fdepth{split}.npy") # center object
    render_pose_val  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
    st_depth_val     = -fdepth

    

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
    val_idx = [72, 76, 80, 140, 144, 148, 208, 212, 216]
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
    
    def generate_fixed_coords(x_fixed, y_fixed, img_w, img_h, device='cuda'):
        # coord_1 생성: x_fixed를 고정하고 y 값을 -1부터 1까지 img_h 갯수만큼 생성
        coord_1 = torch.tensor([[[[x_fixed, y] for y in torch.linspace(-1, 1, img_h)]]], device=device).permute(0, 2, 1, 3)

        # coord_2 생성: y_fixed를 고정하고 x 값을 -1부터 1까지 img_w 갯수만큼 생성
        coord_2 = torch.tensor([[[[ y_fixed,x] for x in torch.linspace(-1, 1, img_w)]]], device=device)

        return coord_1, coord_2
    

    test1, temp = sampling_val_us(75);
    #breakpoint()

    sampling_functions = {
        "uv": sampling_val_uv,
        "us": sampling_val_us
    }
    
    sampling_val = sampling_functions[opt.decom_dim]
        
    #breakpoint()
    cs ,  img_val = sampling_val(72)
    for i in range(len(cs)):
        print(f"cs[{i}].shape : {cs[i].shape}")
        
    # img_arr = (np.array(img_val.cpu())*255).astype(np.uint8)

    # img = Image.fromarray(img_arr)
    # img.save(f'sampling_val_for_epi_test11111.png')
    # #breakpoint()

    def sampling_random_all(device='cuda'):
        num = np.random.randint(1, round(team1_shape-1))
        
        coord_id1 = torch.randperm(team1_shape, device=device)[:num]
        coord_id2 = torch.randint(0, team2_shape, (round(N_samples/num),), device=device)
        
        coord_id = (coord_id1[:,None] * team2_shape + coord_id2[None, :]).reshape(-1)
        
        # color_whole가 이미 GPU에 있으므로 .to(device)를 추가할 필요가 없음
        sampled_data = color_whole[coord_id,:]
        
        coord_1 = mgrid_team1[None ,coord_id1 , : , :]
        coord_2 = mgrid_team2[None, : ,coord_id2 , :]

        return  [coord_1 ,coord_2], sampled_data
    
    def sampling_random_const(device='cuda'):
        
        coord_id1 = torch.randperm(team1_shape, device=device)[:round(N_samples_dim)]
        coord_id2 = torch.randperm(team2_shape, device=device)[:round(N_samples_dim)]
        
        coord_id = (coord_id1[:,None] * team2_shape + coord_id2[None, :]).reshape(-1)
        
        # color_whole가 이미 GPU에 있으므로 .to(device)를 추가할 필요가 없음
        sampled_data = color_whole[coord_id,:]
        
        coord_1 = mgrid_team1[None ,coord_id1 , : , :]
        coord_2 = mgrid_team2[None, : ,coord_id2 , :]

        return  [coord_1 ,coord_2], sampled_data
    
    sampling_ramdom_functions = {
        "all": sampling_random_all,
        "const": sampling_random_const
    }
    
    sampling_random = sampling_ramdom_functions[opt.sample_type]
    #breakpoint()
    # coords , data  = sampling_random()
    # coords_val , data_val  = sampling_val(100)
    # #breakpoint()
    # coords_val , data_val  = sampling_val_for_epi(72)
    

    # def save_images(array, prefix='image'):
    #     num_images, height, width, _ = array.shape
    #     for i in range(num_images):
    #         # 각 이미지의 RGB 값을 0~255 범위로 변환
    #         img_array = (array[i] * 255).astype(np.uint8)
    #         # 이미지 생성
    #         img = Image.fromarray(img_array)
    #         # 파일명 지정 및 이미지 저장
    #         img.save(f'{prefix}_{i+1}.png')
    #     #uvst_whole = torch.tensor(uvst_whole).cuda() 

    # save_images(color_whole_val)

#    uvst_whole_val = torch.tensor(uvst_whole_val) 
#    color_whole_val = torch.tensor(color_whole_val)


    # model = models.get_INR(
    #                 nonlin=nonlin,
    #                 in_features=4,
    #                 out_features=3, 
    #                 hidden_features=hidden_features,
    #                 hidden_layers=hidden_layers,
    #                 first_omega_0=omega0,
    #                 hidden_omega_0=omega0,
    #                 scale=sigma0,
    #                 pos_encode=posencode,
    #                 sidelength=sidelength,
    #                 wire_tunable=opt.wire_tunable,
    #                 real_gabor=opt.real_gabor)
    #   
    model = models_decom.INR_R2L(feat_per_channel=[2,2],
                    coord_hidden_features = opt.coord_width,
                    coord_hidden_layers= opt.coord_depth, 
                    after_hidden_features= opt.width, 
                    after_hidden_layers = opt.depth, 
                    out_features=3, 
                    after_network_type = "MLP",

                    R = opt.R,
                    skip_connection = opt.skip_connection
                    )

    ckpt_paths = glob.glob(os.path.join(ckpt_path,"*.pth"))
    load_epoch = 0
    

    
    if opt.loadcheckpoint or opt.render_only:
        if len(ckpt_paths) > 0:
            for path in ckpt_paths:
                print(ckpt_path)
                ckpt_id = int(os.path.basename(path).split("ep")[1].split(".")[0])
                load_epoch = max(load_epoch, ckpt_id)
            ckpt_name = f"{ckpt_path}/ep{load_epoch}.pth"
            # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
            print(f"Load weights from {ckpt_name}")
            ckpt = torch.load(ckpt_name)
                
            model.load_state_dict(ckpt)

    print(model)
    # Send model to CUDA
    model.cuda()
    
    

  
    def generate_random_coords(num_samples, img_w, img_h, device='cuda'):
        random_coords = []
        xys = []
        
        for _ in range(num_samples):
            # -1부터 1 사이의 랜덤한 x_fixed와 y_fixed 값을 생성
            x_fixed = torch.rand(1).item() * 2 - 1  # [-1, 1] 범위로 변환
            y_fixed = torch.rand(1).item() * 2 - 1  # [-1, 1] 범위로 변환
            xys.append((x_fixed, y_fixed))
            # generate_fixed_coords 함수를 사용하여 좌표 생성
            coord_1, coord_2 = generate_fixed_coords(x_fixed, y_fixed, img_w, img_h, device)
            random_coords.append((coord_1, coord_2))
            
        
        return random_coords, xys
    
    

# 랜덤한 데이터 100개 생성
    pseudo_path = os.path.join(save_dir , 'pseudo_data')
    #breakpoint()

    if opt.render_only:
        N = 10000  # 샘플링할 개수
        rendering_inputs, xys = generate_random_coords(N, w, h)

        # 동영상 저장 설정
        video_path = 'rendered_video.mp4'
        frame_width = img_w  # 이미지 너비
        frame_height = img_h  # 이미지 높이
        fps = 30  # 초당 프레임 수

    
        # 디버깅을 위한 breakpoint

        # GIF 저장을 위한 이미지 리스트
        frames = []
        os.makedirs(pseudo_path, exist_ok=True)
        i = 0
        # 각 입력 데이터에 대해 이미지 생성 및 GIF에 추가
        tbar = tqdm(range(N))
        for input_data, xy in zip(rendering_inputs, xys):
            # 모델을 사용하여 이미지 생성
            with torch.no_grad():
                pred_color = model(input_data)
            
            # 이미지 형태 변환
            pred_img = pred_color.reshape((frame_height, frame_width, 3)).permute((2, 0, 1))

            # 이미지 파일로 저장 (0~1 범위)
            torchvision.utils.save_image(pred_img, f'{pseudo_path}/output_idx_{i}_{xy[0]}_{xy[1]}.png')
            i += 1
            tbar.update(1)
            # RGB 형식으로 변환 (GIF는 RGB 형식을 사용)
            pred_img_rgb = cv2.cvtColor((pred_img.permute((1, 2, 0)).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

            # 디버깅 출력
            print(f"Frame {i} shape: {pred_img_rgb.shape}, dtype: {pred_img_rgb.dtype}")

            # GIF에 프레임 추가
            frames.append(pred_img_rgb)
        print("rendering done")
        os._exit(0)
        # GIF 파일로 저장
        #imageio.mimsave('output.gif', frames, format='GIF', duration=0.1)
        
    # VideoWriter 객체 해제
   
   
    #print('Number of parameters: ', utils.count_parameters(model))
#    print('Input PSNR: %.2f dB'%utils.psnr(im, im_noisy))
    
    # Create an optimizer
#    optim = torch.optim.Adam(lr=learning_rate*min(1, maxpoints/(H*W)),
#                             params=model.parameters())
    optim = torch.optim.Adam(lr=learning_rate,params=model.parameters())
    
    logger.set_metadata("lr",learning_rate)
    
   
    # lr_before = learning_rate
    # logger.set_metadata("lr_before",lr_before)
    # print(f"lr_before : {lr_before}")
   
    # lr_after = learning_rate
    # logger.set_metadata("lr_after",lr_after)
    # print(f"lr_after : {lr_after}")
        
    # rgb_net_lr = lr_after
    # logger.set_metadata("rgb_net_lr",rgb_net_lr)
    # print(f"rgb_net_lr: {rgb_net_lr}")
        
    # optim = torch.optim.Adam([
    #     {'params': model.coord_input_layer.parameters(), 'lr': learning_rate},
    #     {'params': model.net.parameters(), 'lr': lr_after},
    #     {'params': model.rgb_net.parameters(), 'lr': rgb_net_lr},
    # ])
    # print("rgb#@#$@#$@#$@#$")
    # optim_coord = torch.optim.Adam([
    #     {'params': model.coord_net.parameters(), 'lr': lr_before},
    #     {'params': model.after_network.parameters(), 'lr': lr_before},
    # ])
   
    # Schedule to reduce lr to 0.1 times the initial rate in final epoch
    #scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
    #if (opt.nonlin == 'relu') or (opt.nonlin =='relu_skip'):
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995) 
    scheduler_type = {
        "exp": torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999),
        "linear": LambdaLR(optim, lambda x: 0.1**min(x/niters, 1)),
    }
    
    scheduler = scheduler_type[opt.schdule_type]
    #scheduler_coord = scheduler_type[opt.schdule_type]
    #scheduler_coord = torch.optim.lr_scheduler.ExponentialLR(optim_coord, gamma=0.999)
    

    
 #   x = torch.linspace(-1, 1, W)
 #   y = torch.linspace(-1, 1, H)
    
 #   X, Y = torch.meshgrid(x, y, indexing='xy')
 #   coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    
 #   gt = torch.tensor(im).cuda().reshape(H*W, 3)[None, ...]
 #   gt_noisy = torch.tensor(im_noisy).cuda().reshape(H*W, 3)[None, ...]
    
    mse_array = torch.zeros(niters, device='cuda')
    mse_loss_array = torch.zeros(niters, device='cuda')
    time_array = torch.zeros_like(mse_array)
    
 #   best_mse = torch.tensor(float('inf'))
 #   best_img = None
    
 #   rec = torch.zeros_like(gt)
    
    tbar = tqdm(range(niters))
    init_time = time.time()
    train_size = color_whole.shape[0]
    
    
    # uvst_whole = torch.tensor(uvst_whole).cuda()
    # color_whole = torch.tensor(color_whole).cuda()
    #breakpoint()
    epoch_per_batch = train_size//N_samples
   
    for i in tbar:
        epoch = load_epoch + i + 1
#        indices = torch.randperm(H*W)
        #indices = torch.randperm(train_size)
        if opt.benchmark:
            loop_start = torch.cuda.Event(enable_timing=True)
            loop_end = torch.cuda.Event(enable_timing=True)

            # 반복문 시작 전에 기록
            loop_start.record()
        
        #if True:
        if not ((epoch %test_freq == 0) and opt.benchmark):
            for i in range(epoch_per_batch):

                #breakpoint()
                coords , data  = sampling_random()
                output = model(coords)
               
                loss = ((output - data)**2).mean()

                optim.zero_grad()
                #optim_coord.zero_grad()
                loss.backward()
                optim.step()
                #optim_coord.step()
               
               


        else:
            avg_forward_time = 0
            avg_backward_time = 0
            whole_batch_iter = train_size//maxpoints
            #print(f"whole_batch_iter : {whole_batch_iter}")
            for b_idx in range(0, train_size, maxpoints):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                #breakpoint()
                coords , data  = sampling_random()
                start.record()
                output = model(coords)
                end.record()
                torch.cuda.synchronize()

                loss = ((output - data)**2).mean()

      
                
                avg_forward_time += (start.elapsed_time(end) / whole_batch_iter)

    
            
                start.record()
                optim.zero_grad()
                #optim_coord.zero_grad()
                loss.backward()
                optim.step()
                #optim_coord.step()
                end.record()
                torch.cuda.synchronize()
                avg_backward_time += (start.elapsed_time(end) / whole_batch_iter)
            
        if  ((epoch %test_freq == 0) and opt.benchmark):# 반복문 끝난 후 시간 기록
            loop_end.record()
            torch.cuda.synchronize()
            #logger.set_metadata("per_epoch_whole_time",loop_start.elapsed_time(loop_end))
            #print(f"avg_forward_time : {avg_forward_time}, avg_backward_time :  {avg_backward_time},whole_time : {loop_start.elapsed_time(loop_end)}")
            logger.push_time(avg_forward_time , avg_backward_time ,loop_start.elapsed_time(loop_end),epoch)
        
    #time_array[epoch] = time.time() - init_time

        with torch.no_grad():

            if (epoch % opt.save_ckpt_path == 0) and (epoch != 0):
                #breakpoint()
                cpt_path = ckpt_path + f"/ep{epoch}.pth"
                torch.save(model.state_dict(), cpt_path)
                print(f"save ckpt {epoch} : {cpt_path}")    

            if epoch % test_freq ==0:
                avg_inference_time = 0
                avg_backward_time = 0
                

                #print(f"epoch : {epoch}")
                i = 0
                count = 0
                psnr_arr = []
                val_idx = [72, 76, 80, 140, 144, 148, 208, 212, 216]
                
                val_size = len(val_idx)
                for idx in val_idx:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    #breakpoint()
                    coords , gt_color   = sampling_val(idx)
                    #breakpoint()
                    start.record()
                    pred_color   = model(coords)
                    #temp = temp.reshape(*temp.shape[:-1], 3, -1).sum(-1)
                    #breakpoint()
                    end.record()
                    torch.cuda.synchronize()
                    
                    avg_inference_time += (start.elapsed_time(end) / val_size)

                    pred_img = pred_color.reshape((img_h,img_w,3)).permute((2,0,1))
                    #temp_img = temp.reshape((img_h,img_w,3)).permute((2,0,1))
                    gt_img   = torch.tensor(gt_color).reshape((img_h,img_w,3)).permute((2,0,1))

                    if epoch % test_img_save_freq == 0:
                        torchvision.utils.save_image(pred_img,f"{test_path}/test_{count}.png")
                        #torchvision.utils.save_image(temp_img,f"{test_path}/temp_img_{count}.png")
                        torchvision.utils.save_image(gt_img,f"{test_path}/gt_{count}.png")

                    pred_color = pred_color.cpu().numpy()
                    gt_color = gt_color.cpu().numpy()
                    psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
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


        scheduler.step()
        #scheduler_coord.step()
            
        



if __name__ =="__main__":
    opt = parse_argument()
    
    run(opt)