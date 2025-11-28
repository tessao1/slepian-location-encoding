import torch
from torch import nn
import numpy as np
import pyshtools as pysh
from .utils_cache import HarmonicsCache
from .spherical_harmonics import SphericalHarmonics
from .slepian import Slepian

class SlepianSHHybrid(nn.Module, HarmonicsCache):
    def __init__(self, legendre_polys: int = 10, sh_max_degree: int = None, 
                 harmonics_calculation: str = "analytic"):
        """
        Hybrid positional encoder combining Slepian functions and Spherical Harmonics
        
        Args:
            legendre_polys: Maximum degree for Slepian functions
            sh_max_degree: Maximum degree for low-degree spherical harmonics
            harmonics_calculation: SH calculation method ("analytic", "closed-form", "shtools")
        """
        super(SlepianSHHybrid, self).__init__()
        
        self.legendre_polys = legendre_polys
        self.sh_max_degree = sh_max_degree
        self.harmonics_calculation = harmonics_calculation
        
        self.cache_size = 200000
        self._init_cache(self.cache_size)
              
    # Create Slepian encoder instance
        print("Creating Slepian component for coastal features...")
        self.slepian_encoder = Slepian(
            legendre_polys=self.legendre_polys,
            full_dimension=False)
        self.slepian_dim = self.slepian_encoder.embedding_dim
        shannon_number = self.slepian_dim
        
        print(f"Shannon number from Slepian: {shannon_number}")

        if sh_max_degree is None:
                    self.sh_max_degree = self._calculate_sh_degree_from_shannon(shannon_number)
                    print(f"Auto-calculated sh_max_degree: {self.sh_max_degree}")
        else:
            self.sh_max_degree = sh_max_degree
            print(f"Using provided sh_max_degree: {self.sh_max_degree}")

        # Create Spherical Harmonics encoder instance
        print("Creating Spherical Harmonics component for large-scale features...")
        self.sh_encoder = SphericalHarmonics(
            legendre_polys=self.sh_max_degree, 
            harmonics_calculation=self.harmonics_calculation)
        self.sh_dim = self.sh_encoder.embedding_dim
        
        print("Initializing LayerNorm for each branch...")
        self.norm_slep = nn.LayerNorm(self.slepian_dim)
        self.norm_sh   = nn.LayerNorm(self.sh_dim)
        
        self.embedding_dim = self.slepian_dim + self.sh_dim
                
        print(f"Hybrid encoder initialized:")
        print(f"  Slepian functions: {self.slepian_dim}")
        print(f"  Spherical harmonics: {self.sh_dim}")
        print(f"  Total embedding dim: {self.embedding_dim}")

    def _calculate_sh_degree_from_shannon(self, shannon_number):
        """
        Calculate legendre_polys for SH to match Shannon number dimension
        
        Since SH uses range(L) giving degrees 0 to L-1, this produces L² harmonics.
        To match dimensions, L² ≈ shannon_number (or slightly less)
        
        Args:
            shannon_number: Number of Slepian functions (from Shannon number)
        Returns:
            legendre_polys value for SphericalHarmonics
        """
        # Calculate L such that L² ≤ shannon_number
        L = int(np.sqrt(shannon_number))
        L = max(1, L)
        sh_dim = L ** 2
        
        print(f"Shannon number (Slepian): {shannon_number}")
        print(f"Calculated L for SH: {L}")
        print(f"SH will use degrees: 0 to {L-1}")
        print(f"SH dimension: {sh_dim} harmonics")
        
        if sh_dim > shannon_number and L > 1:
            L -= 1
            sh_dim = L ** 2
            print(f"Adjusted L to {L} → SH dimension: {sh_dim}")
        
        return L

    def forward(self, lonlat):
        """
        Forward pass combining Slepian and SH encodings
        
        Args:
            lonlat: tensor of shape (batch_size, 2) with [lon, lat] coordinates in degrees
        Returns:
            tensor of shape (batch_size, embedding_dim) with hybrid encodings
        """
        batch_size = lonlat.shape[0]
        device = lonlat.device
        
        coord_hashes = self._hash_coordinates(lonlat)
        cached_results, missing_indices = self._get_from_cache(coord_hashes, device)
        
        # Initialize output
        Y = torch.zeros((batch_size, self.embedding_dim), device=device, dtype=torch.float32)
        
        # Output cached results
        for idx, cached_result in enumerate(cached_results):
            if cached_result is not None:
                Y[idx] = cached_result.to(device)
        
        if missing_indices:
            missing_lonlat = lonlat[missing_indices]
            
            # These two lines call the forward methods of the respective encoders
            sh_encoding = self.sh_encoder(missing_lonlat)  # Shape: (n_missing, sh_dim) 
            slepian_encoding = self.slepian_encoder(missing_lonlat)  # Shape: (n_missing, slepian_dim)
            # Normalize each branch
            sh_encoding = self.norm_sh(sh_encoding)
            slepian_encoding = self.norm_slep(slepian_encoding)        

            for i, orig_idx in enumerate(missing_indices):
                combined_encoding = torch.cat([slepian_encoding[i],
                                               sh_encoding[i]
                                               ], dim=0)
                
                Y[orig_idx] = combined_encoding
                self._add_to_cache(coord_hashes[orig_idx], combined_encoding)
        
        return Y

    def get_feature_info(self):
        """Return information about the encoding components"""
        return {
            'slepian_dim': self.slepian_dim,
            'sh_dim': self.sh_dim,
            'total_dim': self.embedding_dim,
            'legendre_polys': self.legendre_polys,
            'sh_max_degree': self.sh_max_degree
        }