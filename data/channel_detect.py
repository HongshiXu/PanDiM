import os
from PIL import Image
import numpy as np

# 统计不同通道数的图像数量
def count_channels_in_images(folder_path):
    channel_count = {}  

    # 遍历文件夹，读取所有以.tif结尾的文件
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith('.tif'):
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path)
            mode = img.mode
            img = np.array(img)    

            if img is None:
                print(f"Unable to open image: {image_file}")
                continue

            # 检查图像的通道数
            channels = img.shape[2]
            if channels in channel_count:
                channel_count[channels] += 1
            else:
                channel_count[channels] = 1

    return channel_count

folder_path = r"E:\code\PanSharpening\PanDataset\WV3_data\test128\ms" 
channel_count = count_channels_in_images(folder_path)

print("统计结果：")
for channels, count in channel_count.items():
    print(f"{channels} 通道的图像有 {count} 张")
