import torch
import matplotlib.pyplot as plt
import numpy as np
import os
#from mpl_toolkits.basemap import Basemap
import pandas as pd
import geopandas as gpd


import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_predictions(
    spatialencoder, 
    bds=[-180, -90, 180, 90], 
    title=None, 
    show=False, 
    savepath=None, 
    plot_points=None, 
    class_idx=None,
    save_globe=True
):
    """
    Plot predictions of a spatial encoder on a regular lon-lat grid using Cartopy.
    """
    device = spatialencoder.device

    # Generate grid
    num_pix_lat = 180
    num_pix_lon = 360
    # Generate grid
    lon = torch.tensor(np.linspace(bds[0], bds[2], num_pix_lon),
                   device=device, dtype=torch.float32)
    lat = torch.tensor(np.linspace(bds[1], bds[3], num_pix_lat),
                   device=device, dtype=torch.float32)
    lons, lats = torch.meshgrid(lon, lat)
    lons, lats = lons.T, lats.T
    lonlats = torch.stack([lons, lats], dim=-1).view(-1, 2)

    param = next(spatialencoder.parameters())
    lonlats = lonlats.to(device=param.device, dtype=param.dtype)


    # Compute predictions
    spatialencoder.eval()
    with torch.no_grad():
        if spatialencoder.regression:
            Y = spatialencoder(lonlats)
        else:
            Y = torch.sigmoid(spatialencoder(lonlats))
    if class_idx is not None:
        Y = Y[:, class_idx].unsqueeze(-1)
    Y = Y.view(num_pix_lat, num_pix_lon, Y.size(-1))

    # If multi-class, take argmax
    y = Y.argmax(-1) if Y.size(-1) > 1 else Y.squeeze(-1)

    # Save globe view
    if savepath is not None and save_globe:
        fig = draw_globe(y, lonlats, plot_points, title)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        file, ext = os.path.splitext(savepath)
        fig.savefig(file + "_globe" + ext, transparent=True, bbox_inches="tight", pad_inches=0)

    # 2D map
    fig = draw_map(y, plot_points, title, bds=bds)


    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        file, ext = os.path.splitext(savepath)
        fig.savefig(file + ext, transparent=True, bbox_inches="tight", pad_inches=0)
    
    if show or savepath is None:
        plt.show(block=False)
        plt.pause(2)   # show for 2 seconds, then continue
    plt.close(fig)
    

def plot_predictions_at_points(
    spatialencoder, 
    lonlats, 
    title=None, 
    show=True, 
    savepath=None, 
    class_idx=None,
    plot_kwargs={},
    lonlatscrs="EPSG:4326",
    plot_crs="EPSG:4326"
):
    """
    Scatter plot predictions at specific lon-lat coordinates using GeoPandas + Cartopy.
    """
    device = spatialencoder.device

    if torch.is_tensor(lonlats):
        lonlats = lonlats.cpu().numpy()

    lonlats_torch = torch.tensor(lonlats, device=device)

    spatialencoder.eval()
    with torch.no_grad():
        if spatialencoder.regression:
            Y = spatialencoder(lonlats_torch)
        else:
            Y = torch.sigmoid(spatialencoder(lonlats_torch))
    if class_idx is not None:
        Y = Y[:, class_idx].unsqueeze(-1)
    y = Y.argmax(-1) if Y.size(-1) > 1 else Y.squeeze(-1)

    # Create GeoDataFrame
    df = pd.DataFrame(lonlats, columns=['longitude', 'latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=lonlatscrs)
    gdf['y'] = y.cpu().numpy() if torch.is_tensor(y) else y

    # Reproject if needed
    if plot_crs != lonlatscrs:
        plot_gdf = gdf.to_crs(plot_crs)
    else:
        plot_gdf = gdf

    # Plot
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_gdf.plot(column='y', ax=ax, **plot_kwargs)

    # Add coastlines
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    if title:
        ax.set_title(title)

    if show or savepath is None:
        plt.show()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, transparent=True, bbox_inches="tight", pad_inches=0)

    return fig

    
def scatter_plot_gdf(
    gdf, 
    plot_key='y', 
    plot_map=True, 
    ax=None, 
    plot_kwargs=None, 
    title=''
):
    """
    Scatter plot a GeoDataFrame on a Cartopy map with optional coastlines and styling.

    Args:
        gdf: GeoDataFrame containing geometry and a column `plot_key` for coloring
        plot_key: column in gdf to use for color
        plot_map: if True, add coastlines
        ax: existing matplotlib axis with Cartopy projection; if None, creates PlateCarree
        plot_kwargs: dict of kwargs passed to GeoDataFrame.plot
        title: plot title
    """
    plot_kwargs = plot_kwargs or {}

    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig = ax.figure

    # Plot points
    gdf.plot(column=plot_key, ax=ax, **plot_kwargs)

     # Add map features
    if plot_map:
        # Add coastlines and borders
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='aqua')

        # **Add shapefile overlay using GeoPandas**
        try:
            coast = gpd.read_file("data/ne_50m_coastline/ne_50m_coastline.shp")
            ax.add_geometries(
                coast.geometry,
                crs=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='black',
                linewidth=0.5
            )
        except FileNotFoundError:
            print("Warning: ne_50m_coastline.shp not found. Skipping shapefile overlay.")

    # Set extent to the data bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_extent([bounds[0]-1, bounds[2]+1, bounds[1]-1, bounds[3]+1], crs=ccrs.PlateCarree())

    # Add title
    if title:
        ax.set_title(title)

    # Remove axes
    ax.axis('off')

    # Optional: add colorbar if column is numeric
    if np.issubdtype(gdf[plot_key].dtype, np.number):
        sm = plt.cm.ScalarMappable(cmap=plot_kwargs.get('cmap', 'viridis'), 
                                   norm=plt.Normalize(vmin=gdf[plot_key].min(), vmax=gdf[plot_key].max()))
        sm._A = []
        fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)

    return fig
  

def draw_map(y, plot_points=None, title=None, bds=[-180, -90, 180, 90]):

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Create lon/lat grid matching y
    nlat, nlon = y.shape[:2]
    lon_grid = np.linspace(bds[0], bds[2], nlon)
    lat_grid = np.linspace(bds[1], bds[3], nlat)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    # Plot filled contours
    cf = ax.pcolormesh(lon2d, lat2d, y.cpu().detach().numpy(),
                       cmap='RdBu_r', vmin=0, vmax=1, shading='auto',
                       transform=ccrs.PlateCarree())

    # Optional points
    if plot_points is not None:
        ax.scatter(plot_points[:,0].cpu(), plot_points[:,1].cpu(),
                   c='red', s=10, transform=ccrs.PlateCarree())

    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if title is not None:
        ax.set_title(title)

    plt.colorbar(cf, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)

    return fig



def draw_globe(y, lonlats, plot_points=None, title=None):
    """
    Draw an orthographic globe using Cartopy.
    """
    if torch.is_tensor(y):
        y = y.squeeze().cpu().detach().numpy()
    if torch.is_tensor(lonlats):
        lonlats = lonlats.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': ccrs.Orthographic(central_longitude=30, central_latitude=45)})
    ax.set_global()

    # Base map features
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='coral')
    ax.add_feature(cfeature.OCEAN, facecolor='aqua')

    # Assuming y is on a regular lat-lon grid matching the globe
    nlat, nlon = y.shape[:2]
    lon_grid = np.linspace(-180, 180, nlon)
    lat_grid = np.linspace(-90, 90, nlat)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    # Filled contour
    cf = ax.contourf(lon2d, lat2d, y, transform=ccrs.PlateCarree(), cmap='RdBu_r', alpha=0.75)
    ax.contour(lon2d, lat2d, y, colors='white', alpha=0.5, transform=ccrs.PlateCarree())

    # Optional points
    if plot_points is not None:
        if torch.is_tensor(plot_points):
            plot_points = plot_points.cpu().numpy()
        ax.scatter(plot_points[:,0], plot_points[:,1], c='red', s=10, transform=ccrs.PlateCarree())

    if title:
        ax.set_title(title)

    plt.colorbar(cf, ax=ax, shrink=0.6)
    return fig

def find_matrix_plot_filename(resultsdir, pe, nn):
    candidates = os.listdir(resultsdir)
    candidates = [f for f in candidates if (f"{pe:1.8}-{nn:1.6}" in f)]
    candidates = [f for f in candidates if ("globe" not in f)]
    candidates = [f for f in candidates if f.endswith('.png')]
    
    if len(candidates) > 1:
        print('found multiple pngs that fit the criteria for plotting:')
        print(candidates)
    elif len(candidates) == 0:
        print(f'could not find a png that fits the crieteria for plotting', end='')
        print(f' for {pe}, {nn} in {resultsdir}')
        return
    return candidates[0]


def plot_result_matrix(resultsdir, positional_encoders, neural_networks, show=False, savepath=None):
    fig, axs_arr = plt.subplots(len(positional_encoders), len(neural_networks), figsize=(16*len(neural_networks), 10*len(positional_encoders)))
    
    if len(positional_encoders) == 1:
        axs_arr = [axs_arr]
    if len(neural_networks) == 1:
        axs_arr = [axs_arr]
        
    for pe, ax_row in zip(positional_encoders, axs_arr):
        for nn, ax in zip(neural_networks, ax_row):
            filename = find_matrix_plot_filename(resultsdir, pe, nn)

            image = plt.imread(os.path.join(resultsdir,filename))
            ax.imshow(image)
            ax.axis("off")

    plt.tight_layout()

    if show:
        plt.show()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0, transparent=True)