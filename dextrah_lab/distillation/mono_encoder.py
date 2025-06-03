from PIL import Image
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F


def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function to compute the output size of a convolution layer.
    
    h_w: Tuple[int, int] - height and width of the input
    kernel_size: int or Tuple[int, int] - size of the convolution kernel
    stride: int or Tuple[int, int] - stride of the convolution
    pad: int or Tuple[int, int] - padding
    dilation: int or Tuple[int, int] - dilation rate
    """
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size
    
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride
    
    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad
    
    h = (h_w[0] + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    w = (h_w[1] + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    return h, w


class CustomCNN(nn.Module):
    def __init__(self, input_height, input_width, device):
        super().__init__()
        self.device = device
        num_channel = 3
        
        # Initial input dimensions
        h, w = input_height, input_width
        
        # Layer 1
        h, w = conv_output_size((h, w), kernel_size=6, stride=2)
        layer1_norm_shape = [16, h, w]
        
        # Layer 2
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer2_norm_shape = [32, h, w]
        
        # Layer 3
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer3_norm_shape = [64, h, w]
        
        # Layer 4
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer4_norm_shape = [128, h, w]
        
        # CNN definition
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer1_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer2_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer3_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer4_norm_shape),  # Dynamically calculated layer norm
        )
        

    def forward(self, x, train_encoder=True):
        cnn_x = self.cnn(x)
        return cnn_x


# def get_standard_transform():
#     transform = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ]
#     transform = transforms.Compose(transform)
#     return transform

def get_standard_transform(device):
    # Pre-create the mean and std tensors on the target device with bf16 dtype
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.bfloat16)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.bfloat16)
    
    # Create a lambda transform that explicitly casts to bf16 and normalizes
    transform = [
        transforms.Lambda(lambda x: (x.to(dtype=torch.bfloat16) - mean[None, :, None, None]) / std[None, :, None, None])
    ]
    transform = transforms.Compose(transform)
    return transform




class ResnetEncoder(nn.Module):
    def __init__(self, input_height, input_width, device="cuda", train_resnet=True):
        super().__init__()
        self.device = device

        self.train_resnet = train_resnet

        device = "cuda:0"
        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        ).to(torch.bfloat16)
        # remove last 2 layers of resnet18
        self.resnet18.fc = nn.Identity()
        self.resnet18.avgpool = nn.Identity()

        if train_resnet:
            self.resnet18.train().to(device)
        else:
            self.resnet18.eval().to(device)

        self.transform = get_standard_transform(self.device)

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(40960, 16384)
        )


    def forward(self, x, train_encoder=True):
        x = x.to(torch.bfloat16)

        if train_encoder:
            x = self.transform(x)
            resnet_out = self.resnet18(x)
        else:
            with torch.no_grad():
                x = self.transform(x)
                resnet_out = self.resnet18(x)
        out = self.linear(resnet_out.to(torch.float32))
        return out

class ConvNextEncoder(nn.Module):
    def __init__(self, input_height, input_width, device="cuda", train_resnet=True):
        super().__init__()
        self.device = device

        self.train_resnet = train_resnet

        device = "cuda:0"
        self.convnext = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        ).to(torch.bfloat16)
        # remove last 2 layers of convnext
        self.convnext.avgpool = nn.Identity()
        self.convnext.classifier = nn.Identity()

        if train_resnet:
            self.convnext.train().to(device)
        else:
            self.convnext.eval().to(device)

        self.transform = get_standard_transform(self.device)

        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, 
                      stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Linear layers


    def forward(self, x, train_encoder=True):
        x = x.to(torch.bfloat16)

        if train_encoder:
            x = self.transform(x)
            convnext_out = self.convnext(x)
        else:
            with torch.no_grad():
                x = self.transform(x)
                convnext_out = self.convnext(x)
        out = self.reduce_channels(convnext_out.to(torch.float32))
        return out.reshape(x.shape[0], 128, -1)


MODEL_SETTINGS = {
    "scratch": {
        "n_embd": 128,
        "num_tokens": 234,
        "model": CustomCNN,
    },
    "resnet": {
        "n_embd": 128,
        "num_tokens": 128,
        "model": ResnetEncoder,
    },
    "convnext": {
        "n_embd": 40,
        "num_tokens": 128,
        "model": ConvNextEncoder,
    },
}

class CrossOnlyAttention(nn.Module):
    def __init__(
        self,
        n_embd,               # embedding dimension
        n_head,               # number of attention heads
        attn_pdrop=0.1,       # dropout rate for attention
        resid_pdrop=0.1,      # dropout rate for feed-forward/ residual
        T1=234,               # number of tokens from image 1
        T2=234                # number of tokens from image 2
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.T1 = T1
        self.T2 = T2

        # key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # dropouts
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Precompute the cross-only mask if T1, T2 are fixed
        mask_2d = self.create_cross_attention_mask(T1, T2)  # shape (T, T)
        # shape => (1, 1, T, T) so it broadcasts across (B, n_head, T, T).
        self.register_buffer("cross_mask", mask_2d.view(1, 1, *mask_2d.shape))

    def create_cross_attention_mask(self, T1, T2):
        """
        Returns a tensor of shape (T, T) with 1s for cross-token positions,
        and 0s for same-image positions.
        """
        T = T1 + T2
        img_mask = torch.zeros(T, T)  # shape [T, T]

        # Image 1 attends to Image 2
        img_mask[0:T1, T1:T] = 1
        # Image 2 attends to Image 1
        img_mask[T1:T, 0:T1] = 1

        mask = torch.ones(T+1, T+1)
        mask[1:T+1, 1:T+1] = img_mask


        return mask

    def forward(self, x):
        """
        x shape: (B, T, n_embd), where T = T1 + T2
        """
        B, T, C = x.size()
        head_size = C // self.n_head

        # Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape => (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # (B, n_head, T, head_size) x (B, n_head, head_size, T) => (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))

        # Cross-only mask => disallow attending to same-image tokens
        # att = att.masked_fill(self.cross_mask == 0, float('-inf'))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum over values => (B, n_head, T, head_size)
        y = att @ v

        # Reassemble => (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class SquaredReLU(nn.Module):
    def forward(self, x):
        # ReLU(x) squared
        return F.relu(x).pow(2)


class KeypointModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_tokens, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CrossOnlyAttention(
            n_embd, n_head, attn_pdrop,
            resid_pdrop, T1=n_tokens, T2=n_tokens
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, in_dim, out_dim, ctx_len, n_embd, n_head, num_layer, attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ctx_len = ctx_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.num_layer = num_layer

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.weight_pos_embed = nn.Embedding(ctx_len, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd, n_head, ctx_len,
                    attn_pdrop, resid_pdrop
                )
                for _ in range(num_layer)
            ],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_dim),
        )
        self.embd_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        # (B, T, in_dim)
        x = self.input_layer(x)
        # pose embeds for left and right images
        pos_embeds = self.weight_pos_embed(
            torch.arange(self.ctx_len, device=x.device)
        ).unsqueeze(0)
        x = x + pos_embeds
        x = torch.cat([self.embd_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class MonoEncoder(nn.Module):
    def __init__(
        self, backbone, img_height, img_width, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.backbone = backbone
        self.cnn = MODEL_SETTINGS[backbone]["model"](img_height, img_width, "cuda")
        self.num_tokens = MODEL_SETTINGS[backbone]["num_tokens"]
        if n_embd is None:
            n_embd = MODEL_SETTINGS[backbone]["n_embd"]
        self.out_embd = n_embd # 8
        self.transformer = Transformer(n_embd, self.out_embd, self.num_tokens, n_embd, n_head, 2)
        self.n_embd = n_embd
        # self.keypoint_head = KeypointModule("cuda")

        self.out_layer = nn.Sequential(
            nn.Linear(self.out_embd, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        ) # 288, 128, 13, 18

    def forward(self, x, finetune_backbone=True):
        batch_size = x.shape[0]
        x = self.cnn(x, train_encoder=finetune_backbone)
        if self.backbone == "convnext":
            x = x.reshape(1, batch_size, -1, self.n_embd)
            x = x.permute(1, 0, 2, 3)
            x = x.reshape(batch_size, -1, self.n_embd)
        else:
            x = x.view(1, batch_size, self.n_embd, -1)
            x = x.permute(1, 0, 2, 3) # B, 2, 128, -1
            x = x.reshape(batch_size, -1, self.n_embd)
        x = self.transformer(x)
        # x = self.out_layer(x.view(batch_size, -1))
        x = self.out_layer(x[:, 0, :])
        # return x, kpt_left, kpt_right
        return x


def main():
    im_left = Image.open("left_img.png")
    batch_size = 144

    img_tensor_left = torch.tensor(np.array(im_left)).permute(2, 0, 1).unsqueeze(0) / 255.
    imgs = img_tensor_left.repeat(batch_size, 1, 1, 1).to("cuda")
    backbone = "convnext"
    mono_encoder = MonoEncoder(
        backbone=backbone,
        img_height=240, img_width=320,
        n_embd=MODEL_SETTINGS[backbone]["n_embd"], n_head=4
    ).to("cuda")
    out = mono_encoder(imgs)
    breakpoint()

if __name__ == "__main__":
    main()
