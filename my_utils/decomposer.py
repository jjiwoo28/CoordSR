import torch
import numpy as np
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
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
    """
    Generates a flattened grid of (x, y) coordinates in a range of -1 to 1 for 2D,
    reshaped to the form [1, n, 1, 2] or [1, 1, m, 2].
    """
    if isinstance(sidelen, int):
        sidelen = (sidelen, sidelen)

    # 2차원 그리드 좌표 생성
    pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1).astype(np.float32)
    pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)

    # 좌표를 [-1, 1] 범위로 변환
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
    """
    Generate two separate 2D grids.
    """
    grid_uv = get_mgrid_2d(sidelen1, dim=2)
    grid_st = get_mgrid_2d(sidelen2, dim=1)
    return grid_uv, grid_st

def combine_grids(grid_uv, grid_st):
    """
    Combine two grids to form a new grid with shape [1, n, m, 4].
    """
    n, m = grid_uv.shape[1], grid_st.shape[2]
    # Combine grids
    combined_grid = torch.cat((grid_uv.expand(-1, -1, m, -1), grid_st.expand(-1, n, -1, -1)), dim=-1)
    return combined_grid

# 사용 예제
if __name__ == "__main__":
    sidelen1 = (512,512)  # 첫 번째 2D 그리드의 크기
    sidelen2 = (1, 1)   # 두 번째 2D 그리드의 크기

    grid_uv, grid_st = generate_2d_grids(sidelen1, sidelen2)

    # 두 그리드 결합
    combined_grid = combine_grids(grid_uv, grid_st)

    # 결과 출력
    print("첫 번째 2D 그리드 (u, v) 형태:")
    print(f"그리드 형태: {grid_uv.shape}\n")

    print("두 번째 2D 그리드 (s, t) 형태:")
    print(f"그리드 형태: {grid_st.shape}\n")

    print("결합된 그리드 형태:")
    print(f"그리드 형태: {combined_grid.shape}\n")
