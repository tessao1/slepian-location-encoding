import torch
from torch import nn
import pyshtools as pysh
import numpy as np

def SH(m, l, phi, theta):
    """
    Calculates spherical harmonics using pyshtools
    Args:
        m: order
        l: degree
        phi: azimuthal angle in radians
        theta: polar angle in radians
    Returns:
        torch tensor of spherical harmonics values
    """
    phi_np = phi.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()
 
    y = pysh.expand.spharm_lm(l, m, theta_np, phi_np, normalization='ortho', degrees=False)
        
    y = torch.from_numpy(y).to(phi.device)
        
    return y