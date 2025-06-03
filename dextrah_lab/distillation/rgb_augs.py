from PIL import Image
import random
import os
import glob
import numpy as np

import torch
from torchvision import transforms
import warp as wp


@wp.kernel
def modify_saturation_kernel(
    # inputs
    rgb: wp.array(dtype=float, ndim=4),
    gray: wp.array(dtype=float, ndim=3),
    saturation: wp.array(dtype=float, ndim=1),
    max_pixels: wp.array(dtype=float, ndim=2),
    # outputs
    rgb_out: wp.array(dtype=float, ndim=4)
):
    batch_index, pixel_row, pixel_column = wp.tid()
    r = rgb[batch_index, 0, pixel_row, pixel_column]
    g = rgb[batch_index, 1, pixel_row, pixel_column]
    b = rgb[batch_index, 2, pixel_row, pixel_column]

    gray_val = gray[batch_index, pixel_row, pixel_column]
    max_r = max_pixels[batch_index, 0]
    max_g = max_pixels[batch_index, 1]
    max_b = max_pixels[batch_index, 2]

    rgb_out[batch_index, 0, pixel_row, pixel_column] = (
        gray_val + saturation[batch_index] * (r - gray_val)
    )
    rgb_out[batch_index, 1, pixel_row, pixel_column] = (
        gray_val + saturation[batch_index] * (g - gray_val)
    )
    rgb_out[batch_index, 2, pixel_row, pixel_column] = (
        gray_val + saturation[batch_index] * (b - gray_val)
    )

    rgb_out[batch_index, 0, pixel_row, pixel_column] = wp.clamp(
        rgb_out[batch_index, 0, pixel_row, pixel_column],
        0.,
        max_r
    )
    rgb_out[batch_index, 1, pixel_row, pixel_column] = wp.clamp(
        rgb_out[batch_index, 1, pixel_row, pixel_column],
        0.,
        max_g
    )
    rgb_out[batch_index, 2, pixel_row, pixel_column] = wp.clamp(
        rgb_out[batch_index, 2, pixel_row, pixel_column],
        0.,
        max_b
    )

@wp.kernel
def modify_contrast_kernel(
    # inputs
    rgb: wp.array(dtype=float, ndim=4),
    gray: wp.array(dtype=float, ndim=3),
    avg_brightness: wp.array(dtype=float, ndim=1),
    contrast: wp.array(dtype=float, ndim=1),
    max_pixels: wp.array(dtype=float, ndim=2),
    # outputs
    rgb_out: wp.array(dtype=float, ndim=4)
):
    batch_index, pixel_row, pixel_column = wp.tid()
    r = rgb[batch_index, 0, pixel_row, pixel_column]
    g = rgb[batch_index, 1, pixel_row, pixel_column]
    b = rgb[batch_index, 2, pixel_row, pixel_column]

    gray_val = gray[batch_index, pixel_row, pixel_column]
    contrast_val = contrast[batch_index]

    rgb_out[batch_index, 0, pixel_row, pixel_column] = (
        avg_brightness[batch_index] + contrast_val * (r - avg_brightness[batch_index])  
    )
    rgb_out[batch_index, 1, pixel_row, pixel_column] = (
        avg_brightness[batch_index] + contrast_val * (g - avg_brightness[batch_index])
    )
    rgb_out[batch_index, 2, pixel_row, pixel_column] = (
        avg_brightness[batch_index] + contrast_val * (b - avg_brightness[batch_index])
    )

    rgb_out[batch_index, 0, pixel_row, pixel_column] = min(
        max_pixels[batch_index, 0], max(0., rgb_out[batch_index, 0, pixel_row, pixel_column])
    )
    rgb_out[batch_index, 1, pixel_row, pixel_column] = min(
        max_pixels[batch_index, 1], max(0., rgb_out[batch_index, 1, pixel_row, pixel_column])
    )
    rgb_out[batch_index, 2, pixel_row, pixel_column] = min(
        max_pixels[batch_index, 2], max(0., rgb_out[batch_index, 2, pixel_row, pixel_column])
    )

@wp.kernel
def modify_brightness_kernel(
    # inputs
    rgb: wp.array(dtype=float, ndim=4),
    brightness: wp.array(dtype=float, ndim=1),
    max_pixels: wp.array(dtype=float, ndim=2),
    # outputs
    rgb_out: wp.array(dtype=float, ndim=4)
):
    batch_index, pixel_row, pixel_column = wp.tid()
    r = rgb[batch_index, 0, pixel_row, pixel_column]
    g = rgb[batch_index, 1, pixel_row, pixel_column]
    b = rgb[batch_index, 2, pixel_row, pixel_column]

    rgb_out[batch_index, 0, pixel_row, pixel_column] = min(
        max_pixels[batch_index, 0], max(0., r * brightness[batch_index])
    )
    rgb_out[batch_index, 1, pixel_row, pixel_column] = min(
        max_pixels[batch_index, 1], max(0., g * brightness[batch_index])
    )
    rgb_out[batch_index, 2, pixel_row, pixel_column] = min(
        max_pixels[batch_index, 2], max(0., b * brightness[batch_index])
    )

@wp.kernel
def modify_hue_kernel(
    # inputs
    h: wp.array(dtype=float, ndim=3),
    hue: wp.array(dtype=float, ndim=1),
    # outputs
    h_out: wp.array(dtype=float, ndim=3)
):
    batch_index, pixel_row, pixel_column = wp.tid()
    new_h = h[batch_index, pixel_row, pixel_column] + hue[batch_index]
    if new_h >= 1.0:
        new_h -= 1.0
    elif new_h < 0.0:
        new_h += 1.0
    h_out[batch_index, pixel_row, pixel_column] = new_h


@wp.kernel
def conv2d(
    # inputs
    rgb: wp.array(dtype=float, ndim=4),
    kernel: wp.array(dtype=float, ndim=3),
    kernel_hf_width: int,
    alpha: float,
    in_width: int,
    in_height: int,
    # outputs
    rgb_out: wp.array(dtype=float, ndim=4),
):
    batch_index, y, x = wp.tid()

    sum_r = float(0)
    sum_g = float(0)
    sum_b = float(0)

    for kx in range(-kernel_hf_width, kernel_hf_width+1):
        for ky in range(-kernel_hf_width, kernel_hf_width+1):

            x_idx = wp.clamp(x + kx, 0, in_width-1)
            y_idx = wp.clamp(y + ky, 0, in_height-1)

            kx_idx = kx + kernel_hf_width
            ky_idx = ky + kernel_hf_width

            sum_r = sum_r + float(rgb[batch_index, 0, y_idx, x_idx]) * kernel[0, ky_idx, kx_idx]
            sum_g = sum_g + float(rgb[batch_index, 1, y_idx, x_idx]) * kernel[0, ky_idx, kx_idx]
            sum_b = sum_b + float(rgb[batch_index, 2, y_idx, x_idx]) * kernel[0, ky_idx, kx_idx]

    rgb_out[batch_index, 0, y, x] = alpha*wp.clamp(sum_r, 0.0, 1.0) + (1.0-alpha)*wp.float(rgb[batch_index, 0, y, x])
    rgb_out[batch_index, 1, y, x] = alpha*wp.clamp(sum_g, 0.0, 1.0) + (1.0-alpha)*wp.float(rgb[batch_index, 1, y, x])
    rgb_out[batch_index, 2, y, x] = alpha*wp.clamp(sum_b, 0.0, 1.0) + (1.0-alpha)*wp.float(rgb[batch_index, 2, y, x])


def angle_to_rotation_matrix(angle):

    ang_rad = np.deg2rad(angle)
    cos_a = np.cos(ang_rad)
    sin_a = np.sin(ang_rad)

    rot_mat = np.zeros((2, 2))

    rot_mat[0, 0] = cos_a
    rot_mat[0, 1] = sin_a

    rot_mat[1, 0] = -sin_a
    rot_mat[1, 1] = cos_a

    return rot_mat


def get_rotation_matrix2d(center, angle, scale):

    rotation_matrix = angle_to_rotation_matrix(angle)
    scaling_matrix = np.zeros((2, 2))
    np.fill_diagonal(scaling_matrix, 1.0)

    scaling_matrix = scaling_matrix * scale

    scaled_rotation = rotation_matrix @ scaling_matrix

    alpha = scaled_rotation[0, 0]
    beta = scaled_rotation[0, 1]

    x = center[0]
    y = center[1]

    M = np.eye(3)

    M[0:2, 0:2] = scaled_rotation
    M[0, 2] = (1 - alpha) * x - beta * y
    M[1, 2] = beta * x + (1 - alpha) * y

    return M


def rotate(kernel, angle):

    center = [(kernel.shape[1] - 1)/2, (kernel.shape[0] - 1)/2]

    scale  = 1.0 
    rot_matrix = get_rotation_matrix2d(center, angle, scale)

    x = np.linspace(0, kernel.shape[1]-1, kernel.shape[1])
    y = np.linspace(0, kernel.shape[0]-1, kernel.shape[0])

    xx, yy = np.meshgrid(x, y)

    XY = np.array([xx.flatten(), yy.flatten()])
    XY = np.vstack([XY, np.ones((1, XY.shape[-1]))])

    Rot_XY = np.matmul(np.linalg.inv(rot_matrix), XY)

    out_kernel = np.zeros_like(kernel)

    for idx in range(0, Rot_XY.shape[-1]):

        rotx, roty = Rot_XY[0][idx], Rot_XY[1][idx]

        rotx = max(min(int(rotx+0.5), kernel.shape[1]-1), 0)
        roty = max(min(int(roty+0.5), kernel.shape[0]-1), 0)

        interp_val = kernel[int(roty)][int(rotx)]

        x_org = XY[0][idx]
        y_org = XY[1][idx]

        out_kernel[int(y_org)][int(x_org)] = interp_val
    
    return out_kernel


def get_motion_blur_kernel2d(kernel_size, angle, direction):
    kernel_tuple = (kernel_size, kernel_size)

    # direction from [-1, 1] to [0, 1] range
    direction = (np.clip(direction, -1., 1.) + 1.) / 2.
    kernel = np.zeros((1, kernel_tuple[0], kernel_tuple[1]))

    # Element-wise linspace
    kernel[:, kernel_tuple[0] // 2, :] = np.stack([
        (direction + ((1 - 2 * direction) / (kernel_size - 1)) * i)
        for i in range(kernel_size)
    ], axis=-1)

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel[0], angle)

    kernel = kernel / np.sum(kernel, axis=(0, 1))

    return kernel


def get_motion_blur_kernel2d_batched(batch_size, kernel_size, angle_range, direction_range):
    # Generate random angles and directions for each batch
    angles = np.random.uniform(angle_range[0], angle_range[1], size=(batch_size,))
    directions = np.random.uniform(direction_range[0], direction_range[1], size=(batch_size,))

    # direction from [-1, 1] to [0, 1] range
    directions = (np.clip(directions, -1., 1.) + 1.) / 2.

    # Create base kernels
    kernels = np.zeros((batch_size, kernel_size, kernel_size))

    for i in range(batch_size):
        kernels[i, kernel_size // 2, :] = np.linspace(directions[i], 1 - directions[i], kernel_size)

    # Generate rotation matrices
    angles_rad = angles  # Assuming input is already in radians
    cos_angles = np.cos(angles_rad)
    sin_angles = np.sin(angles_rad)
    rotation_matrices = np.array([
        [cos_angles, -sin_angles],
        [sin_angles, cos_angles]
    ]).transpose(2, 0, 1)

    # Create coordinate grid
    coords = np.stack(np.meshgrid(
        np.arange(kernel_size) - kernel_size // 2,
        np.arange(kernel_size) - kernel_size // 2
    ), axis=-1).reshape(-1, 2).T

    # Rotate coordinates
    rotated_coords = np.einsum('bij,jk->bik', rotation_matrices, coords)

    # Interpolate values
    x_coords = rotated_coords[:, 0, :].reshape(batch_size, kernel_size, kernel_size) + kernel_size // 2
    y_coords = rotated_coords[:, 1, :].reshape(batch_size, kernel_size, kernel_size) + kernel_size // 2

    x0 = np.floor(x_coords).astype(int).clip(0, kernel_size - 1)
    x1 = np.ceil(x_coords).astype(int).clip(0, kernel_size - 1)
    y0 = np.floor(y_coords).astype(int).clip(0, kernel_size - 1)
    y1 = np.ceil(y_coords).astype(int).clip(0, kernel_size - 1)

    x_diff = x_coords - x0
    y_diff = y_coords - y0

    rotated_kernels = (
        kernels[np.arange(batch_size)[:, None, None], y0, x0] * (1 - x_diff) * (1 - y_diff) +
        kernels[np.arange(batch_size)[:, None, None], y0, x1] * x_diff * (1 - y_diff) +
        kernels[np.arange(batch_size)[:, None, None], y1, x0] * (1 - x_diff) * y_diff +
        kernels[np.arange(batch_size)[:, None, None], y1, x1] * x_diff * y_diff
    )

    # Normalize kernels
    rotated_kernels /= rotated_kernels.sum(axis=(1, 2), keepdims=True)

    return rotated_kernels


class RgbAug:
    def __init__(
            self, device, all_env_inds, use_stereo,
            background_cfg, color_cfg, motion_blur_cfg
    ):
        self.device = device
        self.all_env_inds = all_env_inds
        self.num_envs = len(self.all_env_inds)
        self.background_cfg = background_cfg
        self.color_cfg = color_cfg
        self.motion_blur_cfg = motion_blur_cfg

        img_names = glob.glob(os.path.join(background_cfg["dir"], "*.jpg"))#[:self.num_envs]
        self.background_imgs = [
            Image.open(img_name).convert("RGB")
            for img_name in img_names
        ]
        self.background_img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.use_stereo = use_stereo
        self.has_background_aug = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.use_stereo:
            self.env_left_backgrounds = torch.zeros(
                (
                    self.num_envs, 3,
                    background_cfg["height"],
                    background_cfg["width"]
                ),
                dtype=torch.float32, device=self.device
            )
            self.env_right_backgrounds = torch.zeros(
                (
                    self.num_envs, 3,
                    background_cfg["height"],
                    background_cfg["width"]
                ),
                dtype=torch.float32, device=self.device
            )
        else:
            self.env_backgrounds = torch.zeros(
                (
                    self.num_envs, 3,
                    background_cfg["height"],
                    background_cfg["width"]
                ),
                dtype=torch.float32, device=self.device
            )

        self.has_color_aug = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.color_aug_params = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )

        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_inds

        # sample random background images
        p = torch.rand(len(env_ids))
        w_background_aug = p < self.background_cfg["aug_prob"]
        env_ids_with_background_aug = env_ids[w_background_aug].flatten()
        env_ids_without_background_aug = env_ids[~w_background_aug].flatten()
        self.has_background_aug[env_ids_with_background_aug] = True
        self.has_background_aug[env_ids_without_background_aug] = False

        if torch.any(w_background_aug):
            selected_imgs = random.sample(
                self.background_imgs,
                w_background_aug.sum().item()
            )
            background_tensors = torch.stack([
                self.background_img_transform(img)
                for img in selected_imgs
            ]).to(self.device)

            top = torch.randint(
                0, 480 - self.background_cfg["height"] + 1, (1,)
            ).item()
            left = torch.randint(
                0, 640 - self.background_cfg["width"] + 1, (1,)
            ).item()

            if self.use_stereo:
                left_background_img = background_tensors[
                    :, :,
                    top:top+self.background_cfg["height"],
                    left:left+self.background_cfg["width"]
                ]
                self.env_left_backgrounds[
                    env_ids_with_background_aug
                ] = left_background_img
                top = torch.randint(
                    0, 480 - self.background_cfg["height"] + 1, (1,)
                ).item()
                left = torch.randint(
                    0, 640 - self.background_cfg["width"] + 1, (1,)
                ).item()

                right_background_img = background_tensors[
                    :, :,
                    top:top+self.background_cfg["height"],
                    left:left+self.background_cfg["width"]
                ]
                self.env_right_backgrounds[
                    env_ids_with_background_aug
                ] = right_background_img
            else:
                background_img = background_tensors[
                    :, :,
                    top:top+self.background_cfg["height"],
                    left:left+self.background_cfg["width"]
                ]
                self.env_backgrounds[
                    env_ids_with_background_aug
                ] = background_img

        # sample random color jitter params
        p = torch.rand(len(env_ids))
        w_color_aug = p < self.color_cfg["aug_prob"]
        env_ids_with_color_aug = env_ids[w_color_aug].flatten()
        env_ids_without_color_aug = env_ids[~w_color_aug]
        self.has_color_aug[env_ids_with_color_aug] = True
        self.has_color_aug[env_ids_without_color_aug] = False   

        if torch.any(w_color_aug):
            try:
                self.color_aug_params[env_ids_with_color_aug, 0] = torch.rand(
                    env_ids_with_color_aug.shape[0]
                ).to(self.device) * (
                    self.color_cfg["saturation_range"][1] -
                    self.color_cfg["saturation_range"][0]
                ) + self.color_cfg["saturation_range"][0]
            except:
                breakpoint()
            self.color_aug_params[env_ids_with_color_aug, 1] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["contrast_range"][1] -
                self.color_cfg["contrast_range"][0]
            ) + self.color_cfg["contrast_range"][0]
            self.color_aug_params[env_ids_with_color_aug, 2] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["brightness_range"][1] -
                self.color_cfg["brightness_range"][0]
            ) + self.color_cfg["brightness_range"][0]
            self.color_aug_params[env_ids_with_color_aug, 3] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["hue_range"][1] -
                self.color_cfg["hue_range"][0]
            ) + self.color_cfg["hue_range"][0]

    def apply_background_aug(self, rgb_imgs, masks, env_backgrounds):
        if torch.any(self.has_background_aug):
            rgb_imgs[self.has_background_aug] = torch.where(
                masks[self.has_background_aug].expand(-1, 3, -1, -1),
                env_backgrounds[self.has_background_aug],
                rgb_imgs[self.has_background_aug]
            )
        return rgb_imgs
    
    # ripped from pytorch source code
    def _rgb_to_hsv(self, image: torch.Tensor) -> torch.Tensor:
        r, g, _ = image.unbind(dim=-3)

        # Implementation is based on
        # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
        minc, maxc = torch.aminmax(image, dim=-3)

        # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
        # from happening in the results, because
        #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
        #   + H channel has division by `(maxc - minc)`.
        #
        # Instead of overwriting NaN afterwards, we just prevent it from occurring so
        # we don't need to deal with it in case we save the NaN in a buffer in
        # backprop, if it is ever supported, but it doesn't hurt to do so.
        eqc = maxc == minc

        channels_range = maxc - minc
        # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine.
        ones = torch.ones_like(maxc)
        s = channels_range / torch.where(eqc, ones, maxc)
        # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
        # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
        # would not matter what values `rc`, `gc`, and `bc` have here, and thus
        # replacing denominator with 1 when `eqc` is fine.
        channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
        rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(dim=-3)

        mask_maxc_neq_r = maxc != r
        mask_maxc_eq_g = maxc == g

        hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
        hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
        hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))

        h = hr.add_(hg).add_(hb)
        h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
        return torch.stack((h, s, maxc), dim=-3)
    
    # ripped from pytorch source code
    def _hsv_to_rgb(self, img: torch.Tensor) -> torch.Tensor:
        h, s, v = img.unbind(dim=-3)
        h6 = h.mul(6)
        i = torch.floor(h6)
        f = h6.sub_(i)
        i = i.to(dtype=torch.int32)

        sxf = s * f
        one_minus_s = 1.0 - s
        q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
        t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
        p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
        i.remainder_(6)

        vpqt = torch.stack((v, p, q, t), dim=-3)

        # vpqt -> rgb mapping based on i
        select = torch.tensor([[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long)
        select = select.to(device=img.device, non_blocking=True)

        select = select[:, i]
        if select.ndim > 3:
            # if input.shape is (B, ..., C, H, W) then
            # select.shape is (C, B, ...,  H, W)
            # thus we move C axis to get (B, ..., C, H, W)
            select = select.moveaxis(0, -3)

        return vpqt.gather(-3, select)
    
    def apply_color_aug(self, rgb_imgs):
        if torch.any(self.has_color_aug):
            aug_rgb_img = rgb_imgs[self.has_color_aug]
            b = aug_rgb_img.shape[0]
            h, w = aug_rgb_img.shape[2], aug_rgb_img.shape[3]
            gray_scale = (
                aug_rgb_img[:, 0, :, :] * 0.299 +
                aug_rgb_img[:, 1, :, :] * 0.587 +
                aug_rgb_img[:, 2, :, :] * 0.114
            )
            max_pixels = torch.amax(aug_rgb_img, dim=(-2, -1))
            wp.launch(
                kernel=modify_saturation_kernel,
                dim=[b, h, w],
                inputs=[
                    wp.torch.from_torch(aug_rgb_img),
                    wp.torch.from_torch(gray_scale),
                    wp.torch.from_torch(self.color_aug_params[self.has_color_aug, 0]),
                    wp.torch.from_torch(max_pixels),
                ],
                outputs=[
                    wp.torch.from_torch(aug_rgb_img)
                ]
            )
            gray_scale = (
                aug_rgb_img[:, 0, :, :] * 0.299 +
                aug_rgb_img[:, 1, :, :] * 0.587 +
                aug_rgb_img[:, 2, :, :] * 0.114
            )
            wp.launch(
                kernel=modify_contrast_kernel,
                dim=[b, h, w],
                inputs=[
                    wp.torch.from_torch(aug_rgb_img),
                    wp.torch.from_torch(gray_scale),
                    wp.torch.from_torch(torch.mean(gray_scale, dim=(-2, -1))),
                    wp.torch.from_torch(self.color_aug_params[self.has_color_aug, 1]),
                    wp.torch.from_torch(max_pixels),
                ],
                outputs=[
                    wp.torch.from_torch(aug_rgb_img)
                ]
            )
            gray_scale = (
                aug_rgb_img[:, 0, :, :] * 0.299 +
                aug_rgb_img[:, 1, :, :] * 0.587 +
                aug_rgb_img[:, 2, :, :] * 0.114
            )
            wp.launch(
                kernel=modify_brightness_kernel,
                dim=[b, h, w],
                inputs=[
                    wp.torch.from_torch(aug_rgb_img),
                    wp.torch.from_torch(self.color_aug_params[self.has_color_aug, 2]),
                    wp.torch.from_torch(max_pixels),
                ],
                outputs=[
                    wp.torch.from_torch(aug_rgb_img)
                ]
            )
            hsv = self._rgb_to_hsv(aug_rgb_img)
            hue = hsv[:, 0, :, :]
            s = hsv[:, 1, :, :]
            v = hsv[:, 2, :, :]
            wp.launch(
                kernel=modify_hue_kernel,
                dim=[b, h, w],
                inputs=[
                    wp.torch.from_torch(hue),
                    wp.torch.from_torch(self.color_aug_params[self.has_color_aug, 3]),
                ],
                outputs=[
                    wp.torch.from_torch(hue)
                ]
            )
            aug_rgb_img = self._hsv_to_rgb(torch.stack((hue, s, v), dim=-3))
            rgb_imgs[self.has_color_aug] = aug_rgb_img

        return rgb_imgs

    def apply_motion_blur_aug(self, rgb_imgs):
        p = torch.rand(self.num_envs)
        w_motion_aug = p < self.motion_blur_cfg["aug_prob"]
        env_ids_with_motion_blur_aug = self.all_env_inds[w_motion_aug].flatten()
        if torch.any(w_motion_aug):
            angle = np.random.rand(1)*180.0
            direction = 2*(np.random.rand(1)-0.5)
            kernel_size = np.random.choice(self.motion_blur_cfg["kernel_sizes"])
            kernel = get_motion_blur_kernel2d(kernel_size, angle, direction)
            motion_blur_kernels = kernel.reshape(
                1, kernel_size, kernel_size)
            motion_blur_kernels = torch.from_numpy(
                motion_blur_kernels
            ).to(dtype=torch.float32, device=self.device)
            b = motion_blur_kernels.shape[0]
            h, w = (
                rgb_imgs[env_ids_with_motion_blur_aug].shape[2],
                rgb_imgs[env_ids_with_motion_blur_aug].shape[3]
            )
            aug_imgs = rgb_imgs[env_ids_with_motion_blur_aug]
            wp.launch(
                kernel=conv2d,
                dim=[b, h, w],
                inputs=[
                    wp.torch.from_torch(aug_imgs),
                    wp.torch.from_torch(motion_blur_kernels),
                    kernel_size // 2,
                    0.7, w, h
                ],
                outputs=[
                    wp.torch.from_torch(aug_imgs)
                ]
            )
            rgb_imgs[w_motion_aug] = aug_imgs

        return rgb_imgs

    def apply(self, rgb, mask):
        if self.use_stereo:
            left_img = rgb["left_img"]
            right_img = rgb["right_img"]
            left_mask = mask["left_mask"]   
            right_mask = mask["right_mask"]
            left_img = self.apply_background_aug(
                left_img, left_mask, self.env_left_backgrounds
            )
            right_img = self.apply_background_aug(
                right_img, right_mask, self.env_right_backgrounds
            )
            rgb = {
                "left_img": left_img,
                "right_img": right_img
            }
            rgb["left_img"] = self.apply_color_aug(rgb["left_img"])
            rgb["right_img"] = self.apply_color_aug(rgb["right_img"])
            rgb["left_img"] = self.apply_motion_blur_aug(rgb["left_img"])
            rgb["right_img"] = self.apply_motion_blur_aug(rgb["right_img"])
        else:
            img = rgb
            mask = mask
            rgb = self.apply_background_aug(img, mask, self.env_backgrounds)
            rgb = self.apply_color_aug(rgb)
            rgb = self.apply_motion_blur_aug(rgb)

        return rgb
