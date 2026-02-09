import cv2
import numpy as np
# from skimage.feature import greycomatrix, greycoprops
from skimage.feature import graycomatrix, graycoprops
# from skimage.feature.texture import greycomatrix, greycoprops


def compute_glcm_contrast(channel, distances=[1], angles=None, levels=256):
    """
    计算单通道图像的水平/垂直/对角 GLCM 的对比度
    """
    if angles is None:
        # 0°(水平), 90°(垂直), 45°和135°(对角)
        angles = [0, np.pi/2, np.pi/4, 3*np.pi/4]

    # 计算 GLCM
    glcm = graycomatrix(channel, 
                        distances=distances, 
                        angles=angles, 
                        levels=levels, 
                        symmetric=True, 
                        normed=True)
    
    # 提取对比度
    contrast = graycoprops(glcm, 'contrast')
    
    # 按论文方法：水平=0°+180°，垂直=90°+270°，对角=45°+135°
    horizontal = contrast[0,0]  # 0°
    vertical   = contrast[0,1]  # 90°
    diagonal   = (contrast[0,2] + contrast[0,3]) / 2.0  # 平均对角方向
    
    avg_contrast = (horizontal + vertical + diagonal) / 3.0
    
    return horizontal, vertical, diagonal, avg_contrast


def process_rgb_image(image_path):
    """
    处理彩色图片：逐通道计算 GLCM 对比度并输出平均值
    """
    # 读取图片 (BGR -> RGB)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 拆分 R, G, B 通道
    channels = cv2.split(img)

    results = {}
    avg_contrasts = []

    for idx, ch in enumerate(['R','G','B']):
        # 确保通道值在 0-255
        channel = channels[idx].astype(np.uint8)
        h, v, d, avg = compute_glcm_contrast(channel)
        results[ch] = {
            "Horizontal": h,
            "Vertical": v,
            "Diagonal": d,
            "Average Contrast": avg
        }
        avg_contrasts.append(avg)

    # 所有通道的平均对比度
    overall_avg_contrast = np.mean(avg_contrasts)

    return results, overall_avg_contrast


# if __name__ == "__main__":
#     image_path = r"D:\python\deep_L\anomaly detection\MuSc-main\data\mvtec_anomaly_detection\transistor\train\good\001.png" # 替换为你的彩色图片路径
#     results, overall_avg_contrast = process_rgb_image(image_path)

#     print("每个通道的对比度结果：")
#     for ch in results:
#         print(f"{ch} 通道: {results[ch]}")

#     print(f"\n整体平均对比度: {overall_avg_contrast:.2f}")
import os

if __name__ == "__main__":
    root_dir = r"D:\python\deep_L\anomaly detection\MuSc-main\data\mvtec_anomaly_detection"
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for category in categories:
        image_path = os.path.join(root_dir, category, "train", "good", "000.png")
        if os.path.exists(image_path):
            results, overall_avg_contrast = process_rgb_image(image_path)
            print(f"\n类别: {category} - 001.png")
            print(f"平均对比度: {overall_avg_contrast:.2f}")
            for ch in results:
                print(f"  {ch} 通道: {results[ch]}")
        else:
            print(f"\n类别: {category} - 001.png 不存在")