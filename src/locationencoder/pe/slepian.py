import torch
from torch import nn
import numpy as np
import pyshtools as pysh
from .utils_cache import HarmonicsCache
from scipy.ndimage import binary_dilation, binary_erosion

class Slepian(nn.Module, HarmonicsCache):
    def __init__(self, legendre_polys: int = 10, full_dimension: bool = False):
        """
        Slepian functions for geographic position encoding
        
        Args:
            legendre_polys: Maximum degree of spherical harmonics
            full_dimension: If True, use full dimension (L+1)^2; 
                            If False, use Shannon number based on coastline area
        """
        super(Slepian, self).__init__()
        self.legendre_polys = legendre_polys
        self.normalization = 'ortho'
        self.cache_size = 200000
        self.full_dimension = full_dimension

        self._init_cache(self.cache_size)
        
        # Initialize Slepian functions
        self._create_localized_slepian()
        
        # Set embedding dimension based on number of well-concentrated functions
        self.embedding_dim = self.slepian_obj.nmax
    

    def _create_topography_mask(self):
        """
        Create a coastline mask based on Earth topography data
        """
        # Load Earth topography
        topo_coeffs = pysh.datasets.Earth.Earth2014.tbi(lmax=300)
        topo = topo_coeffs.expand(extend=False)
        
        # Create land/ocean mask
        mask = topo.data > 0
        
        # Create coastline mask
        dilated = binary_dilation(mask, iterations=4)
        eroded = binary_erosion(mask, iterations=4)
        coastline_mask = dilated ^ eroded
        
        return coastline_mask  


    def _create_localized_slepian(self):
        """
        Create Slepian functions localized to coastal areas
        """
        coastline_mask = self._create_topography_mask()

        window = pysh.SHGrid.from_array(coastline_mask.astype(float))
        self.slepian_obj = pysh.Slepian.from_mask(window, lmax=self.legendre_polys)
        
        if self.full_dimension:
            print(f"Using full dimension: nmax = {self.slepian_obj.nmax}")
            return
        else:
            print(f"Calculating embedding dimension based on coastline area...")
            nlat, nlon = coastline_mask.shape
            lat_step = np.pi / nlat  # latitude step in radians
            lon_step = 2 * np.pi / nlon  # longitude step in radians
    
            # Create latitude weights (cos(latitude) for spherical area)
            lats = np.linspace(np.pi/2, -np.pi/2, nlat)  # π/2 to -π/2 (90° to -90°)
            lat_weights = np.cos(lats)
    
            total_area = 0
            coastline_area = 0
    
            for i in range(nlat):
                for j in range(nlon):
                    # Area of this grid cell
                    cell_area = lat_weights[i] * lat_step * lon_step
                    total_area += cell_area
            
                    if coastline_mask[i, j]:
                        coastline_area += cell_area
    
            area_fraction = coastline_area / total_area  # This should equal 4π for total sphere
    
            L = self.legendre_polys
            shannon_number = int((L + 1)**2 * area_fraction)
    
            self.slepian_obj.nmax = shannon_number

            print(f"Precise area calculation:")
            print(f"Coastline area fraction: {area_fraction:.4f}")
            print(f"Set nmax to Shannon number: {shannon_number}")

        
    def _slepian_point_encoding(self, lat, lon, degrees=True, nmax=None):
        """Compute Slepian encoding for a single point"""
        N = self.slepian_obj.nmax if nmax is None else min(nmax, self.slepian_obj.nmax)
        vals = np.empty(N)
        
        for a in range(N):
            try:
                sh = self.slepian_obj.to_shcoeffs(alpha=a, normalization=self.normalization)
                vals[a] = sh.expand(lat=[lat], lon=[lon], degrees=degrees)[0]
            except Exception as e:
                print(f"Warning: Failed to compute Slepian function {a} at ({lat}, {lon}): {e}")
                vals[a] = 0.0    
        return vals
    
    
    def forward(self, lonlat):
        """
        Forward pass
        Args:
            lonlat: tensor of shape (batch_size, 2) with [lon, lat] coordinates in degrees
        Returns:
            tensor of shape (batch_size, embedding_dim) with Slepian encodings
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
            
            for i, orig_idx in enumerate(missing_indices):
                lon, lat = missing_lonlat[i]
                encoding = self._slepian_point_encoding(
                    lat.item(), lon.item(), degrees=True, nmax=self.embedding_dim)
                encoding_tensor = torch.tensor(encoding, device=device, dtype=torch.float32)
                Y[orig_idx] = encoding_tensor
                self._add_to_cache(coord_hashes[orig_idx], encoding_tensor)
        
        return Y
    
