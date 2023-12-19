import torch
import gc
import os
import yaml
import wandb
import glob
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

from torch.optim import Adam
from datetime import datetime
from PIL import Image
from einops import rearrange, repeat
from collections import OrderedDict


def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, coords, colours, image_size=(256, 256, 3), device="cpu"):
    batch_size = colours.shape[0]

    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    covariance = torch.stack(
        [torch.stack([sigma_x**2, rho*sigma_x*sigma_y], dim=-1),
        torch.stack([rho*sigma_x*sigma_y, sigma_y**2], dim=-1)],
        dim=-2
    )

    # Check for positive semi-definiteness
    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

    # Choosing quite a broad range for the distribution [-5,5] to avoid any clipping
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # Expanding dims for broadcasting
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)

    # Creating a batch-wise meshgrid using broadcasting
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

    xy = torch.stack([xx, yy], dim=-1)
    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))


    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2

    # kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(batch_size * 3, kernel_size, kernel_size)
    # kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(batch_size, 3, kernel_size, kernel_size)

    c = colours.shape[-1]
    kernel_reshaped = kernel_normalized.repeat(1, c, 1).view(batch_size * c, kernel_size, kernel_size)
    kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(batch_size, c, kernel_size, kernel_size)

    # Calculating the padding needed to match the image size
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    # Adding padding to make kernel size equal to the image size
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2,  # padding left and right
               pad_h // 2, pad_h // 2 + pad_h % 2)  # padding top and bottom

    kernel_rgb_padded = torch.nn.functional.pad(kernel_rgb, padding, "constant", 0)

    # Extracting shape information
    b, c, h, w = kernel_rgb_padded.shape

    # Create a batch of 2D affine matrices
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    # Creating grid and performing grid sampling
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1,2,0)

    return final_image


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, window_size=11, size_average=True):
    # Assuming the image is of shape [N, C, H, W]
    (_, _, channel) = img1.size()

    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)


    # Parameters for SSIM
    C1 = 0.01**2
    C2 = 0.03**2

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    SSIM_numerator = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))
    SSIM_denominator = ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    SSIM = SSIM_numerator / SSIM_denominator

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def d_ssim_loss(img1, img2, window_size=11, size_average=True):
    return ssim(img1, img2, window_size, size_average).mean()


def l1loss(pred, target):
    return nn.L1Loss()(pred, target)


def give_required_data(input_coords, image_size):
    # normalising pixel coordinates [-1,1]
    coords = torch.tensor(input_coords / [image_size[0],image_size[1]], device=device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    colour_values = [image_array[coord[1], coord[0]] for coord in input_coords]
    colour_values_np = np.array(colour_values)
    colour_values_tensor = torch.tensor(colour_values_np, device=device).float()

    return colour_values_tensor, coords


def read_codebook(path):
    df = pd.read_excel(path)
    array = df.values   # array(["'Acta2'", 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=object)

    codebook = [
        np.array(x[1:], dtype=np.float32) for x in array
    ]

    return np.stack(codebook, axis=0)   # (181, 15)


def codeloss(pred_code, codebook):
    simi = pred_code @ codebook.T  # (num_samples, 15) @ (15, 181) = (num_samples, 181)

    simi = simi / torch.norm(pred_code, dim=-1, keepdim=True) / torch.norm(codebook, dim=-1, keepdim=True).T

    
    min_dist = 1 - simi.max(dim=-1)[0]  # (num_samples, )

    return min_dist.mean()


if __name__ == "__main__":
    # Read the config.yml file
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract values from the loaded config
    kernal_size = config["kernal_size"]
    image_size = tuple(config["image_size"])
    primary_samples = config["primary_samples"]
    backup_samples = config["backup_samples"]
    image_file_name = config["image_file_name"]

    exp_dir = "exps/2021_08_04-16_00_00"

    # padding = kernal_size // 2
    image_path = image_file_name

    # read 15 images (single channel) and stack them into a 3D tensor (h, w, 15)
    image_paths = glob.glob(os.path.join(image_file_name, '*.png'))
    image_paths.sort()

    images = []

    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image = image.resize((image_size[0],image_size[0]))
        image = np.array(image)
        image = image / 255.0
        images.append(image)
    
    original_array = np.stack(images, axis=-1)  # (h, w, 15) 
    width, height, _ = original_array.shape 

    model_weights = torch.load(os.path.join(exp_dir, 'ckpt_4000.pth'))
    persistent_mask = torch.load(os.path.join(exp_dir, 'mask_4000.pth'))

    save_folder = os.path.join(oos.path.join(exp_dir, 'images'))
    os.makedirs(save_folder, exist_ok=True)

    output = model_weights[persistent_mask]

    batch_size = output.shape[0]
    sigma_x = torch.sigmoid(output[:, 0])
    sigma_y = torch.sigmoid(output[:, 1])
    rho = torch.tanh(output[:, 2])
    alpha = torch.sigmoid(output[:, 3:-2])
    pixel_coords = torch.tanh(output[:, -2:])

    # point positions
    position_images = []
    for idx, recon in enumerate(rearrange(original_array, 'h w c -> c h w')):    # (15, h, w)
        plt.figure()
        recon = repeat(recon, 'h w -> h w 3')
        plt.imshow(recon)

        px = (pixel_coords.data.cpu().numpy()[:, 0] + 1) * 0.5 * image_size[1]
        py = (pixel_coords.data.cpu().numpy()[:, 1] + 1) * 0.5 * image_size[0]
        px = px.astype(np.int16)
        py = py.astype(np.int16)

        px = np.clip(px, 0, image_size[1] - 1)
        py = np.clip(py, 0, image_size[0] - 1)

        plt.scatter(-px, -py, c='red', marker='x', alpha=alpha[:, idx].data.cpu().numpy())
        plt.axis('off')

        plt.savefig(os.path.join(save_folder, f'position_{idx}.png'), bbox_inches='tight')



