import torch
from torch import nn
from modules.bse import BlockEncoder, BlockDecoder
from modules.vae import VAEEncoder, VAEDecoder
from modules.unet import Denoise3DNet

class DiffCraft(nn.Module):
    