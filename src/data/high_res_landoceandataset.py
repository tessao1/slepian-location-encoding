import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely import prepared
import pandas as pd
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import gc

DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

SIMPLIFY_TOLERANCE = 0.005

def sample_uniform_sphere(n, seed=0):
    """Sample points uniformly on sphere"""
    rng = np.random.RandomState(seed)
    u = rng.rand(n)
    v = rng.rand(n)
    lons = 360 * u - 180
    lats = np.degrees(np.arcsin(2 * v - 1))
    return lons, lats

def sample_fibonacci(n, seed=0):
    """Fibonacci spiral sampling"""
    i = np.arange(n)
    phi = np.pi * (3. - np.sqrt(5.))
    lats = np.degrees(np.arcsin(2*(i / n) - 1))
    lons = np.degrees((i * phi) % (2*np.pi) - np.pi)
    return lons, lats

def get_data_points(n=5000, seed=0, cache=True, sampling_method='fibonacci', split='train'):
    """
    Load or generate high-resolution land-ocean dataset
    
    Args:
        n: number of samples
        seed: random seed
        cache: whether to cache/load from cache
        sampling_method: 'fibonacci', 'uniform', or 'sphericaluniform'
        split: 'train', 'test_uniform', 'test_coastline', 'test_island'
    """
    cachefilename = os.path.join(DATA_DIR, f"highres_landocean_{split}_{sampling_method}_{seed}_{int(n/1000)}k.csv")
    
    if os.path.exists(cachefilename) and cache:
        print(f"reading dataset from {cachefilename}. delete file to regenerate...")
        df = pd.read_csv(cachefilename)
        return df
    else:
        print(f"generating {cachefilename}...")
        
        # Download Natural Earth data
        print("Downloading Natural Earth 10m data...")
        land = gpd.read_file("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip")
        minor_islands = gpd.read_file("https://naciscdn.org/naturalearth/10m/physical/ne_10m_minor_islands.zip")
        coastlines = gpd.read_file("https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip")
        
        print("Processing geometries...")
        
        # Combine land and islands
        all_land = pd.concat([land, minor_islands], ignore_index=True)
        all_land = gpd.GeoDataFrame(all_land, crs=land.crs)
        
        # Compute areas for island classification
        all_land_proj = all_land.to_crs('+proj=moll')
        all_land['area_sq_miles'] = all_land_proj.geometry.area / 1e6 * 0.386102
        
        island_threshold = 30000  # sq miles, from FAIR-EARTH paper
        all_land['is_island'] = all_land['area_sq_miles'] < island_threshold
        
        island_gdf = all_land[all_land['is_island']].copy()
        
        # Simplify geometries
        land_union = unary_union(all_land.geometry).simplify(SIMPLIFY_TOLERANCE)
        coastline_union = unary_union(coastlines.geometry).simplify(SIMPLIFY_TOLERANCE)
        island_union = unary_union(island_gdf.geometry).simplify(SIMPLIFY_TOLERANCE)
        
        land_prep = prepared.prep(land_union)
        island_prep = prepared.prep(island_union)
        
        del land, minor_islands, coastlines, all_land_proj
        gc.collect()
        
        # Helper function to classify points
        def classify_points(lons, lats):
            """Classify points as land/ocean and island/continent"""
            labels = []
            is_island = []
            
            for lon, lat in zip(lons, lats):
                p = Point(lon, lat)
                if land_prep.contains(p):
                    labels.append(1)
                    is_island.append(1 if island_prep.contains(p) else 0)
                else:
                    labels.append(0)
                    is_island.append(0)
            
            return np.array(labels), np.array(is_island)
        
        # Helper function for sampling near geometry
        def sample_near_geometry(geometry, n_samples, buffer_dist=0.5, max_attempts=50):
            """Sample points near a geometry"""
            bounds = geometry.bounds  # minx, miny, maxx, maxy
            buffered = geometry.buffer(buffer_dist)
            buffered_prep = prepared.prep(buffered)
            
            lons, lats = [], []
            attempts = 0
            
            rng = np.random.RandomState(seed)
            while len(lons) < n_samples and attempts < max_attempts:
                # Oversample
                batch_size = (n_samples - len(lons)) * 10
                candidate_lons = rng.uniform(bounds[0] - buffer_dist, bounds[2] + buffer_dist, batch_size)
                candidate_lats = rng.uniform(bounds[1] - buffer_dist, bounds[3] + buffer_dist, batch_size)
                
                # Filter to near geometry
                for lon, lat in zip(candidate_lons, candidate_lats):
                    if len(lons) >= n_samples:
                        break
                    if -90 <= lat <= 90 and buffered_prep.contains(Point(lon, lat)):
                        lons.append(lon)
                        lats.append(lat)
                
                attempts += 1
            
            return np.array(lons[:n_samples]), np.array(lats[:n_samples])
        
        # Generate points based on split and sampling method
        if split == 'train':
            print(f"\nGenerating training set...")
            # Mixed sampling: 40% uniform, 30% coastal, 30% island
            n_uniform = int(n * 0.4)
            n_coast = int(n * 0.3)
            n_island = n - n_uniform - n_coast
            
            print(f"  Uniform: {n_uniform}, Coastal: {n_coast}, Island: {n_island}")
            
            # Uniform samples
            if sampling_method == 'fibonacci':
                lons_uni, lats_uni = sample_fibonacci(n_uniform, seed)
            elif sampling_method == 'sphericaluniform':
                lons_uni, lats_uni = sample_uniform_sphere(n_uniform, seed)
            else:  # 'uniform'
                rng = np.random.RandomState(seed)
                lons_uni = (rng.rand(n_uniform) * 360) - 180
                lats_uni = (rng.rand(n_uniform) * 180) - 90
            
            # Coastal samples
            print("  Sampling coastal points...")
            lons_coast, lats_coast = sample_near_geometry(coastline_union, n_coast, buffer_dist=1.0)
            
            # Island samples
            print("  Sampling island points...")
            lons_island, lats_island = sample_near_geometry(island_union, n_island, buffer_dist=1.0)
            
            # Combine
            lons = np.concatenate([lons_uni, lons_coast, lons_island])
            lats = np.concatenate([lats_uni, lats_coast, lats_island])
            
            # Classify
            print("  Classifying...")
            labels, is_island_labels = classify_points(lons, lats)
            
            print(f"  Training set: {len(lons)} points, {labels.sum()} land, {(labels==0).sum()} ocean")
            
        elif split == 'test_uniform':
            print(f"\nGenerating uniform test set...")
            
            if sampling_method == 'fibonacci':
                lons, lats = sample_fibonacci(n, seed)
            elif sampling_method == 'sphericaluniform':
                lons, lats = sample_uniform_sphere(n, seed)
            else:  # 'uniform'
                rng = np.random.RandomState(seed)
                lons = (rng.rand(n) * 360) - 180
                lats = (rng.rand(n) * 180) - 90
            
            labels, is_island_labels = classify_points(lons, lats)
            print(f"  Uniform test: {len(lons)} points")
            
        elif split == 'test_coastline':
            print(f"\nGenerating coastline challenge test set...")
            # Sample very close to coastlines
            lons, lats = sample_near_geometry(coastline_union, n, buffer_dist=0.3)
            labels, is_island_labels = classify_points(lons, lats)
            print(f"  Coastline test: {len(lons)} points")
            
        elif split == 'test_island':
            print(f"\nGenerating island challenge test set...")
            # Sample on and very near islands
            lons, lats = sample_near_geometry(island_union, n, buffer_dist=0.5)
            labels, is_island_labels = classify_points(lons, lats)
            print(f"  Island test: {len(lons)} points")
        
        # Create DataFrame
        df = pd.DataFrame({
            'lon': lons,
            'lat': lats,
            'label': labels,
            'is_island': is_island_labels
        })
        
        # Save to CSV
        df.to_csv(cachefilename, index=False)
        print(f"Saved: {cachefilename}")
    
    return df

def get_data(n=5000, seed=0, sampling_method='fibonacci', split='train'):
    """Get data as tensors"""
    df = get_data_points(n, seed, cache=True, sampling_method=sampling_method, split=split)
    
    lon = torch.tensor(df['lon'].values, dtype=torch.float32)
    lat = torch.tensor(df['lat'].values, dtype=torch.float32)
    
    lonlats = torch.stack([lon, lat], dim=1)
    
    land = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(-1)
    
    return lonlats, land

class HighResLandOceanDataModule(pl.LightningDataModule):
    def __init__(self, num_samples=20000, batch_size=256, mode='train', sampling_method='fibonacci'):
        """
        High-resolution land-ocean classification DataModule
        
        Args:
            num_samples: number of samples per dataset
            batch_size: batch size for dataloaders
            mode: 'train' or 'tune' (affects validation seed)
            sampling_method: 'fibonacci', 'uniform', or 'sphericaluniform'
        """
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.mode = mode
        self.sampling_method = sampling_method

    def setup(self, stage: str):
        # Training dataset with mixed sampling
        self.train_ds = TensorDataset(*get_data(
            self.num_samples, 
            seed=0, 
            sampling_method=self.sampling_method,
            split='train'
        ))
        
        # Validation: uniform sampling
        val_seed = 1 if self.mode == 'tune' else 2
        self.valid_ds = TensorDataset(*get_data(
            self.num_samples,
            seed=val_seed,
            sampling_method=self.sampling_method,
            split='train'
        ))
        
        # Test datasets
        self.test_uniform_ds = TensorDataset(*get_data(
            self.num_samples,
            seed=3,
            sampling_method=self.sampling_method,
            split='test_uniform'
        ))
        
        self.test_coastline_ds = TensorDataset(*get_data(
            self.num_samples,
            seed=4,
            sampling_method=self.sampling_method,
            split='test_coastline'
        ))
        
        self.test_island_ds = TensorDataset(*get_data(
            self.num_samples,
            seed=5,
            sampling_method=self.sampling_method,
            split='test_island'
        ))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        # Return list of test dataloaders
        return [
            DataLoader(self.test_uniform_ds, batch_size=self.batch_size, shuffle=False),
            DataLoader(self.test_coastline_ds, batch_size=self.batch_size, shuffle=False),
            DataLoader(self.test_island_ds, batch_size=self.batch_size, shuffle=False)
        ]


if __name__ == '__main__':
    # Test data generation
    print("Testing high-resolution land-ocean dataset generation...")
    
    # Generate all datasets
    for split in ['train', 'test_uniform', 'test_coastline', 'test_island']:
        df = get_data_points(n=1000, seed=42, cache=False, sampling_method='fibonacci', split=split)
        print(f"\n{split}: {len(df)} points")
        print(f"  Land: {df['label'].sum()}, Ocean: {(df['label']==0).sum()}")
        print(f"  Islands: {df['is_island'].sum()}")