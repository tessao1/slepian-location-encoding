import torch
from torch import nn
import numpy as np
import pyshtools as pysh
from .utils_cache import HarmonicsCache
from .utils_mask import CoastlineMask

class Slepian(nn.Module, HarmonicsCache):
    def __init__(self, legendre_polys: int = 10):
        """
        Slepian functions for geographic position encoding
        
        Args:
            legendre_polys: Maximum degree of spherical harmonics
            full_dimension: If True, use full dimension (L+1)^2; 
                            If False, use Shannon number based on coastline area
        """
        super().__init__()
        self.legendre_polys = legendre_polys
        self.cache_size = 500000
        self._init_cache(self.cache_size)

        self._create_localized_slepian()
        self.num_modes = int(round(self.slepian.shannon))
        print(f"Slepian modes (Shannon number): {self.num_modes}")
        self.embedding_dim = self.num_modes
        self.coeffs = [
            self.slepian.to_shcoeffs(alpha=a)
            for a in range(self.num_modes)
        ]
        
    def _create_localized_slepian(self):
        """Create Slepian functions localized to the coastlines"""
        mask_dict, nlat, nlon = CoastlineMask.get_mask(self.legendre_polys)
        coastline_mask = mask_dict[self.legendre_polys]
        self.coastline_mask = coastline_mask
        self.mask_nlat = nlat
        self.mask_nlon = nlon

        self.slepian = pysh.Slepian.from_mask(self.coastline_mask, lmax=self.legendre_polys)
    
    def forward(self, lonlat):
        """
        Args:
            lonlat: (N, 2) tensor with [lon, lat] in degrees
        
        Returns:
            (N, num_modes) tensor of Slepian features
        """
        lon = lonlat[:, 0].detach().cpu().numpy()
        lat = lonlat[:, 1].detach().cpu().numpy()
        lon_360 = np.where(lon < 0.0, lon + 360.0, lon)
        
        coord_hashes = self._hash_coordinates(lonlat)
        cached, missing = self._get_from_cache(coord_hashes, lonlat.device)
        
        results = [None] * len(coord_hashes)
        
        if missing:
            lon_m = lon_360[missing]
            lat_m = lat[missing]
            
            computed = np.empty((len(missing), self.num_modes), dtype=np.float32)
            
            for i, (lo, la) in enumerate(zip(lon_m, lat_m)):
                for k, sh in enumerate(self.coeffs):
                    computed[i, k] = sh.expand(lon=float(lo), lat=float(la), degrees=True)
            
            computed = torch.from_numpy(computed).to(lonlat.device)
            
            for j, idx in enumerate(missing):
                self._add_to_cache(coord_hashes[idx], computed[j])
                results[idx] = computed[j]
        
        for i, val in enumerate(cached):
            if val is not None:
                results[i] = val.to(lonlat.device)
        
        return torch.stack(results, dim=0)
 

        
