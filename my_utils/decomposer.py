import torch
import numpy as np

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2  #

def generate_fixed_coords(x_fixed, y_fixed, img_w, img_h, scale, device='cpu'):
    
    # ������ x, y ��ǥ�� �������� ��ǥ �ټ��� �����մϴ�.
    
    # Args:
    #     x_fixed (float): ������ x ��ǥ �� (-1 ~ 1)
    #     y_fixed (float): ������ y ��ǥ �� (-1 ~ 1)
    #     img_w (int): �̹��� �ʺ�
    #     img_h (int): �̹��� ����
    #     scale (int): �ٿ���ø� ������
    #     device (str): �ټ��� ������ ����̽�
        
    # Returns:
    #     tuple: (coord_1, coord_2) ��ǥ �ټ�
    
    coord_1 = torch.tensor([[[[x_fixed, y] for y in torch.linspace(-1, 1, img_h//scale)]]], device=device).permute(0, 2, 1, 3)
    coord_2 = torch.tensor([[[[ y_fixed,x] for x in torch.linspace(-1, 1, img_w//scale)]]], device=device)
    return coord_1, coord_2


def generate_fixed_coords_fully(x_fixed, y_fixed,img_w, img_h, scale, device='cpu'):
    
    # 4���� ��ǥ �׸��� ����
    # Args:
    #     fixed_values: ������ �� �� [value1, value2]
    #     img_w: �̹��� �ʺ�
    #     img_h: �̹��� ����
    #     scale: ������ ����
    #     device: ���� ��ġbreakpoint()
    # Returns:
    #     (1, img_h, img_w, 4) ������ ��ǥ �ټ�
    
    # y, x ��ǥ �׸��� ����
    y = torch.linspace(-1, 1, img_h//scale, device=device)
    x = torch.linspace(-1, 1, img_w//scale, device=device)
    
    # 2D �޽��׸��� ����
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    # ������ ������ ������ ũ��� Ȯ��
    fixed_1 = torch.full_like(x_grid, x_fixed)
    fixed_2 = torch.full_like(x_grid, y_fixed)
    
    # (1, H, W, 4) ���·� ����
    coords = torch.stack([x_grid, y_grid, fixed_1, fixed_2], dim=-1)
    #   coords = coords.unsqueeze(0)  # ��ġ ���� �߰�
    
    return coords

def get_limited_files(folder_path, limit=None):
    
    # ���� ���� PNG ���ϵ��� ��������, limit�� ������ ��� �ش� ������ŭ�� ��ȯ�մϴ�.
    
    # Args:
    #     folder_path (str): PNG ���ϵ��� �ִ� ���� ���
    #     limit (int, optional): �ִ� ���� ���� ����. None�� ��� ��� ���� ��ȯ
    
    # Returns:
    #     list: ���ĵ� PNG ���� ���
        # PNG ���ϸ� ���͸�
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    files.sort()  # ���ϸ� ���� ����
    
    # limit�� ������ ��� �ش� ������ŭ�� ��ȯ
    if limit is not None:
        files = files[:limit]
        
    return files

def combine_coordinates(coord_1, coord_2):
    
    # coord_1 (shape: [B, 1, H, 1, 2])�� coord_2 (shape: [B, 1, 1, W, 2])�� 
    # �����Ͽ� shape [B, 1, H, W, 4] �ټ��� �����մϴ�.
    
    # Args:
    #     coord_1: ù ��° ��ǥ �ټ� [B, 1, H, 1, 2]
    #     coord_2: �� ��° ��ǥ �ټ� [B, 1, 1, W, 2]
        
    # Returns:
    #     combined_coords: ���յ� ��ǥ �ټ� [B, 1, H, W, 4]
    
    B, _, H, _, _ = coord_1.shape
    _, _, _, W, _ = coord_2.shape
    
    # coord_1�� [B, 1, H, W, 2] ���·� Ȯ��
    expanded_coord_1 = coord_1.expand(B, 1, H, W, 2)
    
    # coord_2�� [B, 1, H, W, 2] ���·� Ȯ��
    expanded_coord_2 = coord_2.expand(B, 1, H, W, 2)
    
    # �� ��ǥ�� ������ �������� ����
    combined_coords = torch.cat([expanded_coord_1, expanded_coord_2], dim=-1)
    
    return combined_coords

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


class PseudoImageDataset(Dataset):
    # """
    # Ư�� ���� �� PNG ���� �̸����� (idx, x, y)�� �����ϰ�,
    # �ش� �̹����� �ε��Ͽ� (�̹��� �ټ�, (x,y) �ټ�) ���·� ��ȯ�մϴ�.
    # """
    def __init__(self, folder_path, limit=None, scale=None):
        super().__init__()
        self.folder_path = folder_path
        # get_limited_files �Լ��� ����Ͽ� ���� ��� ��������
        self.image_files = get_limited_files(folder_path, limit)
        
        self.scale = scale

        # ù ��° �̹����� �ε��Ͽ� ���̿� �ʺ� ����
        if self.image_files:
            first_image_path = os.path.join(self.folder_path, self.image_files[0])
            with Image.open(first_image_path) as img:
                self.w, self.h = img.size  # width, height ������ �ùٸ��� �Ҵ�
        else:
            self.h, self.w = 0, 0  # �̹����� ���� ��� �⺻�� ����

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # 1) ���� �̸� �Ľ�: output_idx_<index>_<x>_<y>.png
        name_no_ext = file_name.replace('.png', '')
        splitted = name_no_ext.split('_')
        img_idx = int(splitted[2])
        x_coord = float(splitted[3])
        y_coord = float(splitted[4])

        # 2) �̹��� �ҷ�����
        with Image.open(file_path) as img:
            img_array = np.array(img, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        coord_1, coord_2 = generate_fixed_coords(x_coord, y_coord, self.w, self.h, scale=self.scale, device='cpu')
      
        # (�̹���, (x, y)) Ʃ�÷� ��ȯ
        return [coord_1, coord_2], image_tensor

class PseudoImageDataset4D(Dataset):
    # """
    # Ư�� ���� �� PNG ���� �̸����� (idx, x, y)�� �����ϰ�,
    # �ش� �̹����� �ε��Ͽ� (�̹��� �ټ�, (x,y) �ټ�) ���·� ��ȯ�մϴ�.
    # """
    def __init__(self, folder_path, limit=None, scale=None):
        super().__init__()
        self.folder_path = folder_path
        # get_limited_files �Լ��� ����Ͽ� ���� ��� ��������
        self.image_files = get_limited_files(folder_path, limit)
        
        self.scale = scale

        # ù ��° �̹����� �ε��Ͽ� ���̿� �ʺ� ����
        if self.image_files:
            first_image_path = os.path.join(self.folder_path, self.image_files[0])
            with Image.open(first_image_path) as img:
                self.w, self.h = img.size  # ����: width, height ������ �ùٸ��� �Ҵ�
        else:
            self.h, self.w = 0, 0  # �̹����� ���� ��� �⺻�� ����

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # 1) ���� �̸� �Ľ�: output_idx_<index>_<x>_<y>.png
        #    ��: output_idx_0_0.1647700071334839_0.7063174247741699.png
        name_no_ext = file_name.replace('.png', '')
        splitted = name_no_ext.split('_')
        # splitted[0] = 'output'
        # splitted[1] = 'idx'
        # splitted[2] = indexL
        # splitted[3] = x
        # splitted[4] = y
        img_idx = int(splitted[2])
        x_coord = float(splitted[3])
        y_coord = float(splitted[4])

        # 2) �̹��� �ҷ�����
        with Image.open(file_path) as img:
            img_array = np.array(img, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        coord = generate_fixed_coords_fully(x_coord, y_coord, self.w, self.h, scale=self.scale, device='cpu')
        # 3) ��ǥ �ټ� ���� 
        #coords_tensor = torch.tensor([x_coord, y_coord], dtype=torch.float32)

        # (�̹���, (x, y)) Ʃ�÷� ��ȯ
        
        coord = coord.unsqueeze(0)
        return coord, image_tensor  

class PseudoImageDataset4D_after(Dataset):
    # """
    # Ư�� ���� �� PNG ���� �̸����� (idx, x, y)�� �����ϰ�,
    # �ش� �̹����� �ε��Ͽ� (�̹��� �ټ�, (x,y) �ټ�) ���·� ��ȯ�մϴ�.
    # """
    def __init__(self, folder_path, limit=None, scale=None):
        super().__init__()
        self.folder_path = folder_path
        # get_limited_files �Լ��� ����Ͽ� ���� ��� ��������
        self.image_files = get_limited_files(folder_path, limit)
        
        self.scale = scale

        # ù ��° �̹����� �ε��Ͽ� ���̿� �ʺ� ����
        if self.image_files:
            first_image_path = os.path.join(self.folder_path, self.image_files[0])
            with Image.open(first_image_path) as img:
                self.w, self.h = img.size  # ����: width, height ������ �ùٸ��� �Ҵ�
        else:
            self.h, self.w = 0, 0  # �̹����� ���� ��� �⺻�� ����

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # 1) ���� �̸� �Ľ�: output_idx_<index>_<x>_<y>.png
        #    ��: output_idx_0_0.1647700071334839_0.7063174247741699.png
        name_no_ext = file_name.replace('.png', '')
        splitted = name_no_ext.split('_')
        # splitted[0] = 'output'
        # splitted[1] = 'idx'
        # splitted[2] = indexL
        # splitted[3] = x
        # splitted[4] = y
        img_idx = int(splitted[2])
        x_coord = float(splitted[3])
        y_coord = float(splitted[4])

        # 2) �̹��� �ҷ�����
        with Image.open(file_path) as img:
            img_array = np.array(img, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        img_array = downsample_image(img_array, scale=self.scale)
        
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        coord = generate_fixed_coords_fully(x_coord, y_coord, self.w, self.h, scale=self.scale, device='cpu')
        # 3) ��ǥ �ټ� ���� 
        #coords_tensor = torch.tensor([x_coord, y_coord], dtype=torch.float32)

        # (�̹���, (x, y)) Ʃ�÷� ��ȯ
        
        coord = coord.unsqueeze(0)
        return coord, image_tensor  

    
def get_mgrid(sidelen, dim=2):
    # '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    elif dim == 1:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :] = pixel_coords[0, :] / (sidelen[0] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords



def get_mgrid_2d(sidelen, dim=2):
    # """
    # Generates a flattened grid of (x, y) coordinates in a range of -1 to 1 for 2D,
    # reshaped to the form [1, n, 1, 2] or [1, 1, m, 2].
    # """
    if isinstance(sidelen, int):
        sidelen = (sidelen, sidelen)

    # 2차원 그리?�� 좌표 ?��?��
    pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1).astype(np.float32)
    pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)

    # 좌표�? [-1, 1] 범위�? �??��
    pixel_coords -= 0.5
    pixel_coords *= 2.

    if dim == 2:
        # Reshape to [1, n, 1, 2]
        pixel_coords = pixel_coords.reshape(1, sidelen[0] * sidelen[1], 1, 2)
    else:
        # Reshape to [1, 1, m, 2]
        pixel_coords = pixel_coords.reshape(1, 1, sidelen[0] * sidelen[1], 2)

    return torch.Tensor(pixel_coords)

def generate_2d_grids(sidelen1, sidelen2):
    # """
    # Generate two separate 2D grids.
    # """
    grid_uv = get_mgrid_2d(sidelen1, dim=2)
    grid_st = get_mgrid_2d(sidelen2, dim=1)
    return grid_uv, grid_st

def combine_grids(grid_uv, grid_st):
    # """
    # Combine two grids to form a new grid with shape [1, n, m, 4].
    # """
    n, m = grid_uv.shape[1], grid_st.shape[2]
    # Combine grids
    combined_grid = torch.cat((grid_uv.expand(-1, -1, m, -1), grid_st.expand(-1, n, -1, -1)), dim=-1)
    return combined_grid

# ?��?�� ?��?��
if __name__ == "__main__":
    sidelen1 = (512,512)  # �? 번째 2D 그리?��?�� ?���?
    sidelen2 = (1, 1)   # ?�� 번째 2D 그리?��?�� ?���?

    grid_uv, grid_st = generate_2d_grids(sidelen1, sidelen2)

    # ?�� 그리?�� 결합
    combined_grid = combine_grids(grid_uv, grid_st)

    # 결과 출력
    print("�? 번째 2D 그리?�� (u, v) ?��?��:")
    print(f"그리?�� ?��?��: {grid_uv.shape}\n")

    print("?�� 번째 2D 그리?�� (s, t) ?��?��:")
    print(f"그리?�� ?��?��: {grid_st.shape}\n")

    print("결합?�� 그리?�� ?��?��:")
    print(f"그리?�� ?��?��: {combined_grid.shape}\n")
