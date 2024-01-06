import pandas as pd
import numpy as np
from PIL import Image
import os

"""
----------- image to csv --------------
"""


def img2csv():
    # load the csv file
    csv_file = './dataset/fairface025/fairface_label_val.csv'  # 替换为你的CSV文件路径
    data = pd.read_csv(csv_file)

    # 假设图片在 'train/' 文件夹下
    image_folder = './dataset/fairface025/'

    # 为数据帧添加一个新列，用于存储图片像素向量
    data['pixels'] = None

    # 遍历数据帧中的每一行
    for index, row in data.iterrows():
        # 构造图片的完整路径
        image_path = os.path.join(image_folder, row['file'])
        print(image_path)

        # 加载图片
        with Image.open(image_path) as img:
            # 转换为灰度图
            img = img.convert('L')

            # 将图片转换为NumPy数组，然后打平
            image_vector = np.array(img).flatten()

            # 将像素向量添加到数据帧
            data.at[index, 'pixels'] = image_vector.tolist()

    # 保存新的CSV文件
    data.to_csv('./dataset/fairface025/val_reformat.csv', index=False)


"""
----------- csv to image --------------
"""


def csv2img():
    # load the csv
    csv_file = './dataset/fer2013/train.csv'  # 你的CSV文件路径
    data = pd.read_csv(csv_file)

    # 确保输出文件夹存在
    output_folder = './dataset/fer2013/train'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建一个新的DataFrame来保存图片文件名和emotion
    label_data = pd.DataFrame(columns=['file', 'emotion'])

    # 遍历数据帧中的每一行
    for index, row in data.iterrows():
        # 将pixels列的像素值转换为48x48的图像
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, 'L')

        # 构造图片文件名
        image_file = f"{index + 1}.jpg"
        image_path = os.path.join(output_folder, image_file)
        print(image_file,'saved!')

        # 保存图片
        img.save(image_path)

        # 将文件名和emotion添加到新的DataFrame中
        label_data.loc[index] = [image_file, row['emotion']]

    # 保存label.csv文件
    label_data.to_csv('./dataset/fer2013/train_label.csv', index=False)


"""
---------- main function ---------------
"""


def main():
    csv2img()
    # img2csv()


if __name__ == '__main__':
    main()