import numpy as np
import torch
import pyshtools as pysh
from scipy.ndimage import binary_dilation, binary_erosion

class CoastlineMask:
    """Shared coastline mask for local accuracy computation"""
    _instance = None
    _mask = {}
    _nlat = None
    _nlon = None
    
    @classmethod
    def get_mask(cls, L: int = 10):
        """Get or create the coastline mask (singleton pattern)"""
        if L not in cls._mask:
            print("Creating coastline mask for local accuracy...")
            # Load Earth topography
            topo_coeffs = pysh.datasets.Earth.Earth2014.tbi(lmax=300)
            topo = topo_coeffs.expand(extend=False)

            # Create land/ocean mask
            mask = topo.data > 0
            # Create coastline mask
            dilated = binary_dilation(mask, iterations=4)
            eroded = binary_erosion(mask, iterations=4)
            coastline_mask = dilated ^ eroded

            cls._mask[L] = coastline_mask
            cls._nlat, cls._nlon = coastline_mask.shape
        
        return cls._mask, cls._nlat, cls._nlon
    
    @classmethod
    def is_in_masked_region(cls, lonlat):
        """
        Check if coordinates are within the masked (coastline) region
        
        Args:
            lonlat: tensor of shape (batch_size, 2) with [lon, lat] in degrees
        Returns:
            boolean tensor of shape (batch_size,) indicating if each point is in mask
        """
        
        mask, nlat, nlon = cls.get_mask()
        
        lon = lonlat[:, 0].cpu().numpy()
        lat = lonlat[:, 1].cpu().numpy()
        
        # Convert lon/lat to mask indices
        lat_idx = ((90 - lat) / 180 * nlat).astype(int)
        lat_idx = np.clip(lat_idx, 0, nlat - 1)
        
        lon_idx = ((lon + 180) / 360 * nlon).astype(int)
        lon_idx = np.clip(lon_idx, 0, nlon - 1)
        
        in_mask = mask[lat_idx, lon_idx]
        
        return torch.from_numpy(in_mask).bool().to(lonlat.device)
    
    @classmethod
    def visualize_mask(cls, savepath=None):
        """Visualize the coastline mask"""
        import matplotlib.pyplot as plt
        
        mask, nlat, nlon = cls.get_mask()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(mask, cmap='RdBu_r', extent=[-180, 180, -90, 90], aspect='auto')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Coastline Mask Region')
        plt.colorbar(im, ax=ax, label='In Mask')
        
        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches='tight')
            print(f"Mask visualization saved to {savepath}")
        else:
            plt.show()
        plt.close()