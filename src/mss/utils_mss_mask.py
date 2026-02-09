import numpy as np
import geopandas as gpd
import pyshtools as pysh
from shapely.geometry import Point
from typing import Tuple, Dict
from tqdm import tqdm

class ArcticOceanMask:
    """Create and manage Arctic ocean mask for Slepian functions"""
    
    _cache = {}  # Cache masks by (lmax, lat_min)
    
    @classmethod
    def get_mask(cls, lmax: int = 120, lat_min: float = 65.0) -> Tuple[np.ndarray, int, int]:
        """
        Get or create Arctic ocean mask (cached)
        
        Args:
            lmax: Maximum spherical harmonic degree
            lat_min: Minimum latitude for Arctic region (default: 65°N)
            
        Returns:
            mask: (nlat, nlon) binary array where 1 = ocean, 0 = land
            nlat: number of latitude points
            nlon: number of longitude points
        """
        cache_key = (lmax, lat_min)
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        print(f"Creating Arctic ocean mask (lat >= {lat_min}°N, L={lmax})...")
        
        # Load ocean polygons from Natural Earth
        ocean = gpd.read_file("https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip")
        
        # Create grid matching pyshtools DHGrid convention
        nlat = 3 * lmax + 1
        nlon = 2 * nlat - 1  # DHG grid convention
        
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(0, 360, nlon, endpoint=False)
        
        mask = np.zeros((nlat, nlon), dtype=bool)
        
        # Only process Arctic region
        arctic_lat_idx = np.where(lats >= lat_min)[0]
        
        print(f"  Grid size: {nlat} x {nlon}")
        print(f"  Processing {len(arctic_lat_idx)} Arctic latitude rows...")
        
        for i in tqdm(arctic_lat_idx, desc="  Building mask"):
            lat = lats[i]
            for j, lon in enumerate(lons):
                # Convert to -180 to 180 for shapely
                lon_180 = lon if lon <= 180 else lon - 360
                point = Point(lon_180, lat)
                
                if ocean.contains(point).any():
                    mask[i, j] = True
        
        # Print statistics
        n_ocean = mask[arctic_lat_idx, :].sum()
        n_total = len(arctic_lat_idx) * nlon
        ocean_pct = 100 * n_ocean / n_total
        
        print(f"  Arctic ocean points: {n_ocean:,} / {n_total:,} ({ocean_pct:.1f}%)")
        
        cls._cache[cache_key] = (mask, nlat, nlon)
        return mask, nlat, nlon


def visualize_arctic_mask(lmax: int = 120, lat_min: float = 65.0, savepath: str = None):
    """Visualize the Arctic ocean mask"""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    mask, nlat, nlon = ArcticOceanMask.get_mask(lmax=lmax, lat_min=lat_min)
    
    # Convert mask to lat/lon for plotting
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Only plot Arctic region
    arctic_mask_region = lat_grid >= lat_min
    plot_mask = np.where(arctic_mask_region, mask.astype(float), np.nan)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
        central_longitude=0, central_latitude=90
    ))
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
    
    im = ax.pcolormesh(lon_grid, lat_grid, plot_mask, 
                      transform=ccrs.PlateCarree(),
                      cmap='Blues', alpha=0.6, vmin=0, vmax=1,
                      shading='auto')
    
    plt.colorbar(im, ax=ax, label='Ocean (1) / Land (0)', shrink=0.6)
    ax.set_title(f'Arctic Ocean Mask (lat ≥ {lat_min}°N, L={lmax})')
    
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved mask visualization to {savepath}")
    else:
        plt.show()
    
    plt.close()