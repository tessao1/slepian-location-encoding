import torch
from torch import nn
import numpy as np
from .spherical_harmonics_ylm import SH as SH_analytic
from .spherical_harmonics_closed_form import SH as SH_closed_form
from .spherical_harmonics_shtools import SH as SH_shtools
from .utils_cache import HarmonicsCache

class SphericalHarmonics(nn.Module, HarmonicsCache):
    def __init__(self, legendre_polys: int = 10, harmonics_calculation="analytic"):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        cache_size: maximum number of coordinate sets to cache (default 1000)
        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M  #this equals the full spherical harmonics dimension L²
        self.harmonics_calculation = harmonics_calculation

        self._init_cache(50000)

        if harmonics_calculation == "closed-form":
            self.SH = SH_closed_form
        elif harmonics_calculation == "analytic":
            self.SH = SH_analytic
        elif harmonics_calculation == "shtools":
            self.SH = SH_shtools

         
    def forward(self, lonlat):
        """Force CPU computation to avoid GPU numerical issues"""
        original_device = lonlat.device
        
        lonlat_cpu = lonlat.cpu()
        lon, lat = lonlat_cpu[:, 0], lonlat_cpu[:, 1]
        batch_size = lonlat_cpu.shape[0]

        coord_hashes = self._hash_coordinates(lonlat_cpu)
        cached_results, missing_indices = self._get_from_cache(coord_hashes, torch.device('cpu'))

        # Initialize output tensor on CPU
        Y = torch.zeros((batch_size, self.embedding_dim), dtype=torch.float32)  # No device specified = CPU

        # Output cached results
        for idx, cached_result in enumerate(cached_results):
            if cached_result is not None:
                Y[idx] = cached_result  

        if missing_indices:
            missing_lonlat = lonlat_cpu[missing_indices]
            missing_lon = missing_lonlat[:, 0]
            missing_lat = missing_lonlat[:, 1]
            
            phi = torch.deg2rad(missing_lon + 180)
            theta = torch.deg2rad(missing_lat + 90)

            # Compute spherical harmonics for missing coordinates
            Y_missing = []
            for l in range(self.L):
                for m in range(-l, l + 1):
                    y = self.SH(m, l, phi, theta)
                    if isinstance(y, float):
                        y = y * torch.ones_like(phi)
                    Y_missing.append(y)

            Y_missing = torch.stack(Y_missing, dim=-1)

            # Add to output and cache (keep on CPU)
            for i, orig_idx in enumerate(missing_indices):
                Y[orig_idx] = Y_missing[i]
                self._add_to_cache(coord_hashes[orig_idx], Y_missing[i])  # Cache on CPU

        return Y.to(original_device)
    

