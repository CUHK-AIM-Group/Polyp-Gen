from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
from PIL import Image
import os
import lpips
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from scipy.stats import entropy
import argparse


# target_path = "/data/lsy/workspace/DiffAM/sample_imgs/ckp10k/"

def main(target_path):
    

    tgt_imgs_path = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            # 检查文件是否是图像类型
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                tgt_imgs_path.append(file_path)

    # 使用预训练的InceptionV3模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model.cuda()  # 如果有GPU的话

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tgt_imgs = [preprocess(Image.open(img)) for img in tgt_imgs_path]
    tgt_imgs = torch.stack(tgt_imgs)

    # 计算Inception Score

    def inception_score(imgs, inception_model, batch_size=32, splits=1):
        N = len(imgs)
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

        # 获取预测值
        preds = np.zeros((N, 1000))
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batch_size_i = batch.size()[0]

            with torch.no_grad():
                pred = inception_model(batch)[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = pred.cpu().data.numpy()

        # 计算p(y|x)
        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)

        # 计算KL散度
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


    # 计算Inception Score
    inception_score_mean, inception_score_std = inception_score(tgt_imgs, inception_model)
    print("Inception Score: ", inception_score_mean, inception_score_std)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, required=True, help='Path to the target directory')
    args = parser.parse_args()
    main(args.target_path)


#python IS_LPIPS.py --target_path /data/lsy/workspace/DiffAM/sample_imgs/random_mask/ckp30k