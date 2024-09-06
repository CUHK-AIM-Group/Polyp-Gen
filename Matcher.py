import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # 假设x的形状是 (batch_size, sequence_length, feature_dim)
        # 对每个样本的所有序列特征进行GeM pooling
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=1)  # 沿着sequence_length维度取平均
        x = x.pow(1. / self.p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Dinov2Matcher:

  def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=512, half_precision=False, device="cuda"):
      self.repo_name = repo_name
      self.model_name = model_name
      # self.smaller_edge_size = smaller_edge_size
      self.half_precision = half_precision
      self.device = device

      if self.half_precision:
        self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
      else:
        self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

      # self.model = AutoModel.from_pretrained('/data/lsy/workspace/hf_ckp/models--facebook--dinov2-base').to(device)

      self.model.eval()

      self.transform = transforms.Compose([
          # transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
      self.gem_loss = GeM()

  # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
  def prepare_image(self, rgb_image_numpy):
    image = Image.fromarray(rgb_image_numpy)
    image_tensor = self.transform(image)
    resize_scale = image.width / image_tensor.shape[2]

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
    return image_tensor, grid_size, resize_scale

  def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
    cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
    image = Image.fromarray(cropped_mask_image_numpy)
    resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
    resized_mask = np.asarray(resized_mask).flatten()
    return resized_mask

  def extract_global_features(self, image_tensor):
    with torch.inference_mode():  
        if self.half_precision:
            image_batch = image_tensor.unsqueeze(0).half().to(self.device)
        else:
            image_batch = image_tensor.unsqueeze(0).to(self.device)

        tokens = self.model.get_intermediate_layers(image_batch)[0].mean(dim=1).detach().cpu()
        
    return tokens.numpy()

  def extract_local_features(self, image_tensor):
    with torch.inference_mode():
        if self.half_precision:
            image_batch = image_tensor.unsqueeze(0).half().to(self.device)
        else:
            image_batch = image_tensor.unsqueeze(0).to(self.device)

        tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
    return tokens.cpu().numpy()


  def idx_to_source_position(self, idx, grid_size, resize_scale):
    row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    return row, col

