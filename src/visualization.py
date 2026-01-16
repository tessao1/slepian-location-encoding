"""
Camera-ready visualization for land-ocean classification.
Produces publication-quality plots for challenging regions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely import prepared
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# plot style configuration
# -----------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# color scheme
COLORS = {
    'ocean': '#4575b4',      # blue
    'land': '#1a9850',       # green
    'ocean_pred': '#74add1', # light blue
    'land_pred': '#66c2a5',  # light green
    'correct': '#f7f7f7',    # light gray
    'error': '#d73027',      # red
}

CMAP_TRUTH = ListedColormap([COLORS['ocean'], COLORS['land']])
CMAP_PRED = ListedColormap([COLORS['ocean_pred'], COLORS['land_pred']])
CMAP_ERROR = ListedColormap([COLORS['correct'], COLORS['error']])

# -----------------------------
# challenging regions
# -----------------------------
REGIONS = {
    'Indonesia': {
        'bounds': (95, 141, -11, 8),  # lon_min, lon_max, lat_min, lat_max
        'title': 'Indonesian Archipelago',
        'description': 'Complex island chains with varied sizes'
    },
    'Caribbean': {
        'bounds': (-85, -59, 10, 27),
        'title': 'Caribbean Islands',
        'description': 'Scattered small islands'
    },
    'Aegean': {
        'bounds': (22, 30, 35, 42),
        'title': 'Aegean Sea (Greece)',
        'description': 'Dense island clusters with intricate coastlines'
    }
}

# -----------------------------
# data generation for regions
# -----------------------------
def generate_region_grid(region_name, resolution=0.05):
    """Generate dense grid for a specific region"""
    bounds = REGIONS[region_name]['bounds']
    lon_min, lon_max, lat_min, lat_max = bounds
    
    lons = np.arange(lon_min, lon_max, resolution)
    lats = np.arange(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    
    return lon_flat, lat_flat, lon_grid.shape

def get_ground_truth(lons, lats, land_union=None):
    """Get ground truth labels for coordinates"""
    if land_union is None:
        land_union = load_land_geometry()
    
    land_prep = prepared.prep(land_union)
    labels = np.array([1 if land_prep.contains(Point(lon, lat)) else 0 
                       for lon, lat in zip(lons, lats)])
    return labels

def load_land_geometry(simplify_tolerance=0.001):
    """Load and cache land geometry"""
    print("Loading land geometry...")
    land = gpd.read_file("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip")
    minor_islands = gpd.read_file("https://naciscdn.org/naturalearth/10m/physical/ne_10m_minor_islands.zip")
    
    import pandas as pd
    all_land = pd.concat([land, minor_islands], ignore_index=True)
    all_land = gpd.GeoDataFrame(all_land, crs=land.crs)
    
    land_union = unary_union(all_land.geometry).simplify(simplify_tolerance)
    return land_union

def get_predictions(model, lons, lats, device='cuda'):
    """Get model predictions for coordinates"""
    model.eval()
    model.to(device)
    
    lonlat = torch.tensor(np.stack([lons, lats], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        # process in batches to avoid OOM
        batch_size = 10000
        all_probs = []
        
        for i in range(0, len(lonlat), batch_size):
            batch = lonlat[i:i+batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
        
        probs = np.concatenate(all_probs).flatten()
    
    preds = (probs > 0.5).astype(int)
    return preds, probs

# -----------------------------
# single region visualization
# -----------------------------
def plot_region_comparison(model, region_name, land_union=None, 
                           resolution=0.05, device='cuda', 
                           save_path=None, show=True):
    """
    Plot ground truth vs prediction for a single region.
    Returns figure for further customization if needed.
    """
    if region_name not in REGIONS:
        raise ValueError(f"Unknown region: {region_name}. Choose from {list(REGIONS.keys())}")
    
    region = REGIONS[region_name]
    bounds = region['bounds']
    
    # generate grid
    lons, lats, grid_shape = generate_region_grid(region_name, resolution)
    
    # get ground truth
    if land_union is None:
        land_union = load_land_geometry()
    labels = get_ground_truth(lons, lats, land_union)
    
    # get predictions
    preds, probs = get_predictions(model, lons, lats, device)
    
    # reshape for plotting
    labels_grid = labels.reshape(grid_shape)
    preds_grid = preds.reshape(grid_shape)
    probs_grid = probs.reshape(grid_shape)
    errors_grid = (labels != preds).astype(int).reshape(grid_shape)
    
    # compute metrics
    accuracy = (labels == preds).mean()
    
    # create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    
    # ground truth
    ax = axes[0]
    im = ax.imshow(labels_grid, extent=extent, origin='lower', 
                   cmap=CMAP_TRUTH, aspect='auto', interpolation='nearest')
    ax.set_title('Ground Truth')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # prediction
    ax = axes[1]
    ax.imshow(preds_grid, extent=extent, origin='lower',
              cmap=CMAP_TRUTH, aspect='auto', interpolation='nearest')
    ax.set_title(f'Prediction (Acc: {accuracy:.3f})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # error map
    ax = axes[2]
    ax.imshow(errors_grid, extent=extent, origin='lower',
              cmap=CMAP_ERROR, aspect='auto', interpolation='nearest')
    ax.set_title(f'Errors (n={errors_grid.sum()})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ocean'], label='Ocean'),
        mpatches.Patch(facecolor=COLORS['land'], label='Land'),
    ]
    axes[0].legend(handles=legend_elements, loc='lower left', framealpha=0.9)
    
    error_legend = [
        mpatches.Patch(facecolor=COLORS['correct'], edgecolor='gray', label='Correct'),
        mpatches.Patch(facecolor=COLORS['error'], label='Error'),
    ]
    axes[2].legend(handles=error_legend, loc='lower left', framealpha=0.9)
    
    fig.suptitle(region['title'], fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, {'accuracy': accuracy, 'errors': errors_grid.sum(), 
                 'total': labels.size}

# -----------------------------
# multi-region comparison
# -----------------------------
def plot_all_regions(model, land_union=None, resolution=0.05, 
                     device='cuda', save_path=None, show=True):
    """
    Plot all challenging regions in a single figure.
    """
    if land_union is None:
        land_union = load_land_geometry()
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    all_metrics = {}
    
    for row, region_name in enumerate(REGIONS.keys()):
        region = REGIONS[region_name]
        bounds = region['bounds']
        extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
        
        # generate data
        lons, lats, grid_shape = generate_region_grid(region_name, resolution)
        labels = get_ground_truth(lons, lats, land_union)
        preds, probs = get_predictions(model, lons, lats, device)
        
        # reshape
        labels_grid = labels.reshape(grid_shape)
        preds_grid = preds.reshape(grid_shape)
        errors_grid = (labels != preds).astype(int).reshape(grid_shape)
        
        accuracy = (labels == preds).mean()
        all_metrics[region_name] = {'accuracy': accuracy, 'errors': errors_grid.sum()}
        
        # ground truth
        ax = axes[row, 0]
        ax.imshow(labels_grid, extent=extent, origin='lower',
                  cmap=CMAP_TRUTH, aspect='auto', interpolation='nearest')
        ax.set_ylabel(region['title'], fontsize=11, fontweight='bold')
        if row == 0:
            ax.set_title('Ground Truth', fontsize=11)
        if row == 2:
            ax.set_xlabel('Longitude')
        
        # prediction
        ax = axes[row, 1]
        ax.imshow(preds_grid, extent=extent, origin='lower',
                  cmap=CMAP_TRUTH, aspect='auto', interpolation='nearest')
        if row == 0:
            ax.set_title('Prediction', fontsize=11)
        if row == 2:
            ax.set_xlabel('Longitude')
        
        # add accuracy annotation
        ax.text(0.02, 0.98, f'Acc: {accuracy:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # error map
        ax = axes[row, 2]
        ax.imshow(errors_grid, extent=extent, origin='lower',
                  cmap=CMAP_ERROR, aspect='auto', interpolation='nearest')
        if row == 0:
            ax.set_title('Error Map', fontsize=11)
        if row == 2:
            ax.set_xlabel('Longitude')
        
        ax.text(0.02, 0.98, f'Errors: {errors_grid.sum()}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # add colorbar/legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ocean'], label='Ocean'),
        mpatches.Patch(facecolor=COLORS['land'], label='Land'),
        mpatches.Patch(facecolor=COLORS['correct'], edgecolor='gray', label='Correct'),
        mpatches.Patch(facecolor=COLORS['error'], label='Error'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, all_metrics

# -----------------------------
# model comparison figure
# -----------------------------
def plot_model_comparison(models_dict, land_union=None, resolution=0.05,
                          device='cuda', save_path=None, show=True):
    """
    Compare multiple models across all regions.
    
    Args:
        models_dict: dict of {model_name: model}
    """
    if land_union is None:
        land_union = load_land_geometry()
    
    n_models = len(models_dict)
    n_regions = len(REGIONS)
    
    fig, axes = plt.subplots(n_regions, n_models + 1, 
                             figsize=(3.5 * (n_models + 1), 3.5 * n_regions))
    
    all_metrics = {name: {} for name in models_dict.keys()}
    
    for row, region_name in enumerate(REGIONS.keys()):
        region = REGIONS[region_name]
        bounds = region['bounds']
        extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
        
        # generate data
        lons, lats, grid_shape = generate_region_grid(region_name, resolution)
        labels = get_ground_truth(lons, lats, land_union)
        labels_grid = labels.reshape(grid_shape)
        
        # ground truth (first column)
        ax = axes[row, 0]
        ax.imshow(labels_grid, extent=extent, origin='lower',
                  cmap=CMAP_TRUTH, aspect='auto', interpolation='nearest')
        ax.set_ylabel(region['title'], fontsize=11, fontweight='bold')
        if row == 0:
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
        if row == n_regions - 1:
            ax.set_xlabel('Longitude')
        
        # each model
        for col, (model_name, model) in enumerate(models_dict.items(), 1):
            preds, probs = get_predictions(model, lons, lats, device)
            preds_grid = preds.reshape(grid_shape)
            accuracy = (labels == preds).mean()
            
            all_metrics[model_name][region_name] = accuracy
            
            ax = axes[row, col]
            ax.imshow(preds_grid, extent=extent, origin='lower',
                      cmap=CMAP_TRUTH, aspect='auto', interpolation='nearest')
            
            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold')
            if row == n_regions - 1:
                ax.set_xlabel('Longitude')
            
            # accuracy annotation
            ax.text(0.02, 0.98, f'{accuracy:.3f}', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ocean'], label='Ocean'),
        mpatches.Patch(facecolor=COLORS['land'], label='Land'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, all_metrics

# -----------------------------
# metrics summary table
# -----------------------------
def print_region_metrics(metrics_dict):
    """Print formatted metrics table"""
    print("\n" + "="*60)
    print("REGIONAL ACCURACY COMPARISON")
    print("="*60)
    
    model_names = list(metrics_dict.keys())
    region_names = list(REGIONS.keys())
    
    # header
    header = f"{'Region':<20}"
    for name in model_names:
        header += f"{name:>15}"
    print(header)
    print("-"*60)
    
    # rows
    for region in region_names:
        row = f"{REGIONS[region]['title']:<20}"
        for model_name in model_names:
            acc = metrics_dict[model_name][region]
            row += f"{acc:>15.4f}"
        print(row)
    
    # averages
    print("-"*60)
    row = f"{'Average':<20}"
    for model_name in model_names:
        avg = np.mean([metrics_dict[model_name][r] for r in region_names])
        row += f"{avg:>15.4f}"
    print(row)
    print("="*60)

# -----------------------------
# quick test
# -----------------------------
if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print(f"Available regions: {list(REGIONS.keys())}")
    print("\nUsage:")
    print("  from visualization import plot_model_comparison, plot_all_regions")
    print("  fig, metrics = plot_model_comparison({'Model A': model_a, 'Model B': model_b})")
