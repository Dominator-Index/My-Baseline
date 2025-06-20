import wandb
import numpy as np
from PIL import Image
from MNIST_config import config, device
import os

# 使用你的 API 密钥登录
wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")

# 在训练过程中记录损失和上传图片等
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])

# 设定图片所在目录
image_dir = "/home/ouyangzl/BaseLine/MNIST/images"

# 获取目录下所有的图片文件
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 遍历每一张图片并上传
for image_file in image_files:
    # 读取图片
    img_path = os.path.join(image_dir, image_file)
    image = Image.open(img_path)
    
    # 使用 wandb.Image 来上传图片
    wandb.log({"image": wandb.Image(image, caption=image_file)})
    
    print(f"Uploaded {image_file} to wandb.")

# 结束 experiment
wandb.finish()