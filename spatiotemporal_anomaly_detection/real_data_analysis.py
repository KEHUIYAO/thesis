import numpy as np
import rasterio
import rasterio.plot
import geopandas
import earthpy.plot
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import earthpy.spatial
import statsmodels.graphics.tsaplots
import statsmodels.tsa.ar_model
import scipy.stats
import statsmodels.formula.api
import sklearn.neighbors
from scipy.spatial.distance import cdist
from anomaly_detection import spatiotemporal_anomaly_detection



def crop_rasterfile_using_shapefile(rasterfile_path, crop_shapefile_path, path_out, show_figures=False):
    """crop the raster file with a shape file.

    Args:
        rasterfile_path (str): file path of the raster file
        crop_shapefile_path (str): file path of the shape file
        path_out (str): the path to save the cropped raster file
        show_figures (bool): if True, show the raster file and the cropped region


    """


    # open the lidar chm
    with rasterio.open(rasterfile_path) as src:
        lidar_chm_im = src.read(masked=True)[0]
        extent = rasterio.plot.plotting_extent(src)
        soap_profile = src.profile

    if show_figures:
        earthpy.plot.plot_bands(lidar_chm_im,
                       cmap='terrain',
                       extent=extent,
                       cbar=False)

    # open crop extent
    crop_extent_soap = geopandas.read_file(crop_shapefile_path)

    if show_figures:
        # plot the two layers together to ensure the overlap each other. If the shapefile does not overlap the raster, then you can not use it to crop!
        fig, ax = plt.subplots(figsize=(10, 10))
        earthpy.plot.plot_bands(lidar_chm_im,
                      cmap='terrain',
                      extent=extent,
                      ax=ax,
                      cbar=False)
        crop_extent_soap.plot(ax=ax, alpha=.6, color='g')
        plt.show()

    # crop the data
    with rasterio.open(rasterfile_path) as src:
        lidar_chm_crop, soap_lidar_meta = earthpy.spatial.crop_image(src, crop_extent_soap)


    # mask the nodata and plot the newly cropped raster layer
    lidar_chm_crop_ma = np.ma.masked_equal(lidar_chm_crop[0], -9999.0)

    if show_figures:
        earthpy.plot.plot_bands(lidar_chm_crop_ma, cmap='terrain', cbar=False)

    # Save to disk so you can use the file later.

    with rasterio.open(path_out, 'w',
                       width=lidar_chm_crop[0].shape[1],
                       height=lidar_chm_crop[0].shape[0],
                       count=1,
                       dtype=lidar_chm_crop[0].dtype,
                       transform=soap_lidar_meta["transform"]) as ff:

        ff.write(lidar_chm_crop[0], 1)

if __name__ == '__main__':
    rasterfile_list_path = ['./Yakutia_EVI_Product/Data/LAEA_y' +
                            str(year) + '_Maximum_Summertime_MODIS_EVI.tif' for year in range(2002, 2022)]

    crop_shapefile_path = './SHP/y2014_Wildfires_Subregions_Yakutia.shp'
    for i in range(len(rasterfile_list_path)):
        rasterfile_path = rasterfile_list_path[i]
        crop_image_path = './image/crop_image' + str(2002+i) + '.tif'
        crop_rasterfile_using_shapefile(rasterfile_path, crop_shapefile_path, path_out=crop_image_path, show_figures=False)

    crop_image_list_path = './image/crop_image'
    crop_image_list_path = [crop_image_list_path + str(year) + '.tif' for year in range(2002, 2022)]
    data = []
    for i in range(len(crop_image_list_path)):
        crop_image_path = crop_image_list_path[i]
        with rasterio.open(crop_image_path) as src:
            data.append(src.read(1))

    data = np.stack(data, axis=-1)

    n_row = data.shape[0]
    n_col = data.shape[1]
    locations = np.array([(i // n_row, i % n_col) for i in range(n_row * n_col)])
    dist = cdist(locations, locations, metric='euclidean')

    # visualize the raw data
    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            ax[i, j].imshow(data[..., ind], cmap='gray_r')
            ax[i, j].axis('off')

            # add year to the title of each subplot, font size is 8
            ax[i, j].set_title(f'{2002 + ind}', fontsize=8)


    plt.show()
    fig.savefig('figure/raw_data.png')

    # NN no laws
    res_list= spatiotemporal_anomaly_detection(data.reshape(n_row*n_col, -1), locations, ts='NN', laws=False, one_sided='right', horizon=1, input_size=1)
    res = res_list[2].reshape(n_row, n_col, -1)

    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            temp = res[..., ind]
            ax[i, j].imshow(temp, cmap='gray_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            # add year to the title of each subplot, font size is 8
            ax[i, j].set_title(f'{2002 + ind}', fontsize=8)


    plt.show()
    fig.savefig(f'figure/nn_no_laws.png')

    # NN laws
    res_list = spatiotemporal_anomaly_detection(data.reshape(n_row * n_col, -1), locations, ts='NN', laws=True,
                                                one_sided='right', horizon=1, input_size=1)
    res = res_list[2].reshape(n_row, n_col, -1)
    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            ax[i, j].imshow(res[..., ind], cmap='gray_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            # add year to the title of each subplot, font size is 8
            ax[i, j].set_title(f'{2002 + ind}', fontsize=8)
    plt.show()
    fig.savefig(f'figure/nn_laws.png')



    # statistical method no laws
    res_list= spatiotemporal_anomaly_detection(data.reshape(n_row*n_col, -1), locations, ts='outlier_test', laws=False, one_sided='right')
    res = res_list[2].reshape(n_row, n_col, -1)

    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            temp = res[..., ind]
            ax[i, j].imshow(temp, cmap='gray_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            # add year to the title of each subplot, font size is 8
            ax[i, j].set_title(f'{2002 + ind}', fontsize=8)

    plt.show()
    fig.savefig(f'figure/statistical_method_no_laws.png')

    # statistical method laws
    res_list = spatiotemporal_anomaly_detection(data.reshape(n_row * n_col, -1), locations, ts='NN', laws=True,
                                                one_sided='right')
    res = res_list[2].reshape(n_row, n_col, -1)
    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            ax[i, j].imshow(res[..., ind], cmap='gray_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            # add year to the title of each subplot, font size is 8
            ax[i, j].set_title(f'{2002 + ind}', fontsize=8)
    plt.show()
    fig.savefig(f'figure/statistical_method_laws.png')
