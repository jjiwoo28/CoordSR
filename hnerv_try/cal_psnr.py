import numpy as np
import cv2
from math import log10

def calculate_psnr(img1_path, img2_path):
    """
    두 이미지 파일의 PSNR(Peak Signal-to-Noise Ratio)을 계산합니다.
    
    Args:
        img1_path (str): 첫 번째 이미지 파일 경로
        img2_path (str): 두 번째 이미지 파일 경로(비교 대상)
        
    Returns:
        float: PSNR 값 (dB)
    """
    # 이미지 읽기
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 이미지 크기가 같은지 확인
    if img1.shape != img2.shape:
        raise ValueError("두 이미지의 크기가 다릅니다.")
    
    # MSE(Mean Squared Error) 계산
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    # MSE가 0이면 두 이미지가 완전히 동일함
    if mse == 0:
        return float('inf')
    
    # 8비트 이미지의 최대 픽셀 값은 255
    max_pixel = 255.0
    
    # PSNR 계산
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    
    return psnr

# 사용 예시
if __name__ == "__main__":
    # 이미지 경로
    original_img = "output/0331/knights_teacher_8_256/1_1_1_pe_2_16_Dim64_16_FC2_2_KS0_1_5_RED1.2_low32_blk1_1_e1600_b1_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext__DEC_pshuffel_4,4,4,2,2_relu1_1/gt_0000.png"
    compared_img = "output/0331/knights_teacher_8_256/1_1_1_pe_2_16_Dim64_16_FC2_2_KS0_1_5_RED1.2_low32_blk1_1_e1600_b1_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext__DEC_pshuffel_4,4,4,2,2_relu1_1/out_0006.png"
    
    # PSNR 계산
    psnr_value = calculate_psnr(original_img, compared_img)
    print(f"PSNR 값: {psnr_value} dB")