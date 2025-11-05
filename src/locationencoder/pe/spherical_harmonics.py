import torch
from torch import nn
import numpy as np
from .spherical_harmonics_ylm import SH as SH_analytic
from .spherical_harmonics_closed_form import SH as SH_closed_form
from .spherical_harmonics_shtools import SH as SH_shtools
from .utils_cache import HarmonicsCache

class SphericalHarmonics(nn.Module, HarmonicsCache):
    def __init__(self, legendre_polys: int = 10, harmonics_calculation="analytic", cache_size=2000):
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
        self.embedding_dim = self.L * self.M
        self.harmonics_calculation = harmonics_calculation

        self._init_cache(cache_size)

        if harmonics_calculation == "closed-form":
            self.SH = SH_closed_form
        elif harmonics_calculation == "analytic":
            self.SH = SH_analytic
        elif harmonics_calculation == "shtools":
            self.SH = SH_shtools

        
    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]
        batch_size = lonlat.shape[0]
        device = lonlat.device

        # Create hash for each coordinate pair
        coord_hashes = self._hash_coordinates(lonlat)
        cached_results, missing_indices = self._get_from_cache(coord_hashes, device)

        # Initialize output tensor
        Y = torch.zeros((batch_size, self.embedding_dim), device=device, dtype=torch.float32)

        # Fill in cached results
        for idx, cached_result in enumerate(cached_results):
            if cached_result is not None:
                Y[idx] = cached_result.to(device)

        # Compute missing results
        if missing_indices:
            # Get coordinates that need computation
            missing_lonlat = lonlat[missing_indices]
            missing_lon = missing_lonlat[:, 0]
            missing_lat = missing_lonlat[:, 1]
            
             # Convert degree to rad
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

            # Add to output and cache
            for i, orig_idx in enumerate(missing_indices):
                Y[orig_idx] = Y_missing[i]
                self._add_to_cache(coord_hashes[orig_idx], Y_missing[i])

        return Y

    

