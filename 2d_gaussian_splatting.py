import torch
import gc
import os
import yaml
import wandb
import glob
import tqdm
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


def generate_2D_gaussian_splatting(
        kernel_size, 
        sigma_x, 
        sigma_y, 
        rho, 
        coords, 
        colours, 
        image_size=(256, 256, 3), 
        device="cpu"
    ):
    batch_size = colours.shape[0]

    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    # rho = torch.zeros(batch_size, 1, 1, device=device)

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

    # init wandb
    wandb.init(
        project="mrna_2dgs",
        entity="star6",
        config=config,
        name=config["run_name"],
    )

    # Extract values from the loaded config
    kernal_size = config["kernal_size"]
    image_size = list(config["image_size"])
    primary_samples = config["primary_samples"]
    backup_samples = config["backup_samples"]
    num_epochs = config["num_epochs"]
    densification_interval = config["densification_interval"]
    learning_rate = config["learning_rate"]
    image_file_name = config["image_file_name"]
    display_interval = config["display_interval"]
    grad_threshold = config["gradient_threshold"]
    gauss_threshold = config["gaussian_threshold"]
    display_loss = config["display_loss"]
    gpu_id = config["gpu_id"]
    codebook_path = config["codebook_path"]
    report_interval = config["report_interval"]
    checkpoint_interval = config["checkpoint_interval"]
    w_ssim = config["w_ssim"]
    w_code = config["w_code"]
    w_circle = config["w_circle"]

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Read the codebook
    codebook = read_codebook(codebook_path) # (181, 15) numpy array cpu
    codebook = torch.tensor(codebook, device=device)    # tensor 

    # prepare points
    num_samples = primary_samples + backup_samples

    padding = kernal_size // 2
    image_path = image_file_name

    # read 15 images (single channel) and stack them into a 3D tensor (h, w, 15)
    image_paths = [os.path.join(image_file_name, f'F1_R{r}_Ch{c}.png') for r in range(1, 6) for c in range(2, 5)] 
    image_paths.sort()

    images = []

    for image_path in image_paths:
        image = Image.open(image_path).convert('L')

        ori_size = image.size

        image_size[0] = ori_size[0] if ori_size[0] < image_size[0] else image_size[0]
        image_size[1] = ori_size[1] if ori_size[1] < image_size[1] else image_size[1]

        image = image.resize((image_size[0],image_size[1]))
        image = np.array(image)
        image = image / 65536
        images.append(image)
    
    original_array = np.stack(images, axis=-1)  # (h, w, 15)

    print('max value:', original_array.max(), 'min value:', original_array.min())
    original_array = (original_array - original_array.min()) / (original_array.max() * 1.6 - original_array.min())

    width, height, _ = original_array.shape  

    image_array = original_array
    target_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)
    coords = np.random.randint(0, [width, height], size=(num_samples, 2))
    random_pixel_means = torch.tensor(coords, device=device)
    pixels = [image_array[coord[0], coord[1]] for coord in coords]
    pixels_np = np.array(pixels)
    random_pixels =  torch.tensor(pixels_np, device=device)

    colour_values, pixel_coords = give_required_data(coords, image_size)

    pixel_coords = torch.atanh(pixel_coords)

    sigma_values = torch.rand(num_samples, 2, device=device)
    rho_values = 2 * torch.rand(num_samples, 1, device=device) - 1
    alpha_values = torch.ones(num_samples, original_array.shape[-1], device=device) * colour_values
    # W_values = torch.cat([sigma_values, rho_values, alpha_values, colour_values, pixel_coords], dim=1)
    W_values = torch.cat([sigma_values, rho_values, alpha_values, pixel_coords], dim=1) # (num_samples, 19)

    starting_size = primary_samples
    left_over_size = backup_samples
    persistent_mask = torch.cat([torch.ones(starting_size, dtype=bool),torch.zeros(left_over_size, dtype=bool)], dim=0)
    current_marker = starting_size

    # Get current date and time as string
    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Create a directory with the current date and time as its name
    directory = f"./exps/{now}"
    os.makedirs(os.path.join(directory, "checkpoints"), exist_ok=True)

    model_weights = nn.Parameter(W_values)

    optimizer = Adam([model_weights], lr=learning_rate)

    loss_history = []

    """## Training Loop ##"""
    for epoch in tqdm.trange(num_epochs):

        #find indices to remove and update the persistent mask
        if epoch % (densification_interval + 1) == 0 and epoch > 0:
            # if avg alpha is less than 0.01, remove the point
            indices_to_remove = (torch.sigmoid(model_weights[:, 3:-2]).max(dim=-1) < 0.01).nonzero(as_tuple=True)[0] 

            if len(indices_to_remove) > 0:
                print(f"number of pruned points: {len(indices_to_remove)}")

            persistent_mask[indices_to_remove] = False

            # Zero-out parameters and their gradients at every epoch using the persistent mask
            model_weights.data[~persistent_mask] = 0.0


        gc.collect()
        # torch.cuda.empty_cache()

        output = model_weights[persistent_mask]

        batch_size = output.shape[0]

        sigma_x = torch.sigmoid(output[:, 0])
        sigma_y = torch.sigmoid(output[:, 1])
        rho = torch.tanh(output[:, 2])
        alpha = torch.sigmoid(output[:, 3:-2])
        # colours = torch.sigmoid(output[:, 4:7])
        # pixel_coords = torch.tanh(output[:, 7:9])
        pixel_coords = torch.tanh(output[:, -2:])

        # colours_with_alpha = colours * alpha.view(batch_size, 1)
        colours_with_alpha = alpha
        g_tensor_batch = generate_2D_gaussian_splatting(kernal_size, sigma_x, sigma_y, rho, pixel_coords, colours_with_alpha, image_size, device)

        # loss
        l1_loss = l1loss(g_tensor_batch, target_tensor)
        ssim_loss = d_ssim_loss(g_tensor_batch, target_tensor)
        code_loss = codeloss(alpha, codebook)
        circle_loss = torch.nn.functional.mse_loss(sigma_x, sigma_y)

        loss = l1_loss

        if w_ssim > 0:
            loss += w_ssim * ssim_loss
        
        if w_code > 0:
            loss += w_code * code_loss

        if w_circle > 0:
            loss += w_circle * circle_loss

        optimizer.zero_grad()
        loss.backward()

        # Apply zeroing out of gradients at every epoch
        if persistent_mask is not None:
            model_weights.grad.data[~persistent_mask] = 0.0

        if epoch % densification_interval == 0 and epoch > 0:
            # Calculate the norm of gradients
            gradient_norms = torch.norm(model_weights.grad[persistent_mask][:, -2:], dim=1, p=2)
            gaussian_norms = torch.norm(torch.sigmoid(model_weights.data[persistent_mask][:, 0:2]), dim=1, p=2)

            sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
            sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)

            large_gradient_mask = (sorted_grads > grad_threshold)
            large_gradient_indices = sorted_grads_indices[large_gradient_mask]

            large_gauss_mask = (sorted_gauss > gauss_threshold)
            large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

            common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
            common_indices = large_gradient_indices[common_indices_mask]
            distinct_indices = large_gradient_indices[~common_indices_mask]

            # Split points with large coordinate gradient and large gaussian values and descale their gaussian
            if len(common_indices) > 0:
                print(f"number of splitted points: {len(common_indices)}")
                start_index = current_marker + 1
                end_index = current_marker + 1 + len(common_indices)
                persistent_mask[start_index: end_index] = True
                model_weights.data[start_index:end_index, :] = model_weights.data[common_indices, :]
                scale_reduction_factor = 1.6
                model_weights.data[start_index:end_index, 0:2] /= scale_reduction_factor
                model_weights.data[common_indices, 0:2] /= scale_reduction_factor
                current_marker = current_marker + len(common_indices)

            # Clone it points with large coordinate gradient and small gaussian values
            if len(distinct_indices) > 0:
                print(f"number of cloned points: {len(distinct_indices)}")
                start_index = current_marker + 1
                end_index = current_marker + 1 + len(distinct_indices)
                persistent_mask[start_index: end_index] = True
                model_weights.data[start_index:end_index, :] = model_weights.data[distinct_indices, :]
                current_marker = current_marker + len(distinct_indices)

        optimizer.step()

        if epoch % display_interval == 0:
            # point positions
            position_images = []
            for idx, recon in enumerate(target_tensor.data.cpu().permute(2, 0, 1).numpy()):    # (15, h, w)
                plt.figure()
                recon = repeat(recon, 'h w -> h w 3')
                plt.imshow(recon)

                px = (pixel_coords.data.cpu().numpy()[:, 0] + 1) * 0.5 * image_size[1]
                py = (pixel_coords.data.cpu().numpy()[:, 1] + 1) * 0.5 * image_size[0]
                px = px.astype(np.int16)
                py = py.astype(np.int16)

                px = np.clip(px, 0, image_size[1] - 1)
                py = np.clip(py, 0, image_size[0] - 1)

                plt.scatter(-px, py, c='red', marker='x', alpha=alpha[:, idx].data.cpu().numpy())
                plt.axis('off')

                position_images.append(wandb.Image(plt, caption=f"image_{idx}"))
            
            # all_position
            plt.figure()
            px = (pixel_coords.data.cpu().numpy()[:, 0] + 1) * 0.5 * image_size[1]
            py = (pixel_coords.data.cpu().numpy()[:, 1] + 1) * 0.5 * image_size[0]
            px = px.astype(np.int16)
            py = py.astype(np.int16)

            px = np.clip(px, 0, image_size[1] - 1)
            py = np.clip(py, 0, image_size[0] - 1)

            plt.scatter(-px, -py, c='red', marker='x')
            plt.axis('off')

            all_position = wandb.Image(plt, caption="all_position")

            recon = repeat(g_tensor_batch, 'h w c -> c h w 3').data.cpu().numpy()
            gt = repeat(target_tensor, 'h w c -> c h w 3').data.cpu().numpy()
            views = np.stack([recon, gt], axis=1)   # (15, 2, h, w, 3)
            views = views.reshape(-1, *views.shape[2:])

            # logging
            wandb.log({
                "view/recon": [wandb.Image(v, caption=f"recon_gt_{i // 2}") for i, v in enumerate(views)],
                "view/position_images": position_images,
                "view/all_position": all_position,
            }, step=epoch)

        
        if epoch % report_interval == 0:
            wandb.log({
                "train/total_loss": loss.item(),
                "train/l1_loss": l1_loss.item(),
                "train/ssim_loss": ssim_loss.item(),
                "train/code_loss": code_loss.item(),
                "train/circle_loss": circle_loss.item(),
                "params/num_points": len(output),
            }, step=epoch)
            # print(f"epoch: {epoch}, loss: {loss.item()}, num: {len(output)}")

        if epoch % checkpoint_interval == 0 and epoch > 0:
            torch.save(model_weights, os.path.join(directory, 'checkpoints', f"ckpt_{epoch}.pth"))
            torch.save(persistent_mask, os.path.join(directory, 'checkpoints', f"mask_{epoch}.pth"))

