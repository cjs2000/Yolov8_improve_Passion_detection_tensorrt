import os
from PIL import Image
import numpy as np

def calculate_brightness(image_path, threshold=120):
    """
    计算图片的平均亮度并判断是亮度高还是低
    """
    try:
        # 打开图片并转换为灰度图
        img = Image.open(image_path).convert("L")
        # 转换为 numpy 数组
        img_array = np.array(img)
        # 计算亮度平均值
        avg_brightness = np.mean(img_array)
        
        return avg_brightness > threshold
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

def process_folder(folder_path, threshold=120):
    """
    统计文件夹中亮度高与亮度低的图片数量
    """
    high_brightness_count = 0
    low_brightness_count = 0

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".jpg"):
            image_path = os.path.join(folder_path, file_name)
            is_high = calculate_brightness(image_path, threshold)
            if is_high:
                high_brightness_count += 1
            else:
                low_brightness_count += 1

    return high_brightness_count, low_brightness_count

# 测试
folder_path = "image"  # 替换为你的文件夹路径
threshold = 120  # 设置亮度阈值
high, low = process_folder(folder_path, threshold)

print(f"Number of high brightness images: {high}")
print(f"Number of low brightness images: {low}")
