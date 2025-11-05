import torch
import numpy as np
from collections import OrderedDict

class HarmonicsCache:
    """Class to add caching functionality to positional encoders"""
    
    def _init_cache(self, cache_size=2000):
        """Initialize cache - call this in the __init__ method"""
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _hash_coordinates(self, lonlat):
        """
        Create hash from lonlat coordinates for cache lookup
        Args:
            lonlat: tensor of shape (batch_size, 2) with lon, lat coordinates
        Returns:
            tuple of hashes, one per coordinate pair
        """
        # Convert to numpy and create hash for each coordinate pair
        lonlat_np = lonlat.detach().cpu().numpy()
        hashes = []
        for coord in lonlat_np:
            # Round to reduce cache misses from floating point precision
            coord_rounded = tuple(np.round(coord, decimals=6))
            # Create hash from rounded coordinates
            coord_hash = hash(coord_rounded)
            hashes.append(coord_hash)
        return tuple(hashes)

    def _get_from_cache(self, coord_hashes, device):
        """
        Retrieve cached results for given coordinate hashes
        Args:
            coord_hashes: tuple of coordinate hashes
            device: torch device to place results on
        Returns:
            cached_results: list of cached results or None if not found
            missing_indices: indices of coordinates not in cache
        """
        cached_results = []
        missing_indices = []
            
        for idx, coord_hash in enumerate(coord_hashes):
            if coord_hash in self.cache:
                # Move to front (LRU)
                self.cache.move_to_end(coord_hash)
                cached_results.append(self.cache[coord_hash])
                self.cache_hits += 1
            else:
                cached_results.append(None)
                missing_indices.append(idx)
                self.cache_misses += 1
            
        return cached_results, missing_indices

    def _add_to_cache(self, coord_hash, Y_row):
        """
        Add computed result to cache with size limiting
        Args:
            coord_hash: hash of coordinate pair
            Y_row: computed result for this coordinate
        """
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest (FIFO)
        
        # Add new entry
        self.cache[coord_hash] = Y_row.detach().cpu()

    def get_cache_stats(self):
        """Return cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def clear_cache(self):
        """Clear the cache and reset statistics"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0