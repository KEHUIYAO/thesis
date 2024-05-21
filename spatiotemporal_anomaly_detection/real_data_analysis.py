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

def laws_procedure(p_values, pis, weights, alpha):
    """the laws procedure for FDR control.

    Args:
        p_values (array_like): original p_values at each location.
        pis (array_like): sparsity level for each location.
        weights (array_like): adaptive weights at each location.
        alpha (float): threshold to adjust multiplicity.
    Returns:
        array_like: a binary array with the same dimension as the p_values. The position where a signal is detected is equal to 1.
    """

    # calculate weighted p-values
    weighted_p_values = p_values / weights
    # weighted_p_values[weighted_p_values > 1] = 1

    # order the weighted p-values
    weighted_p_values_flattened = weighted_p_values.flatten()  # flatten before sort
    weighted_p_values_flattened_sorted_ind = np.argsort(weighted_p_values_flattened)


    # find the largest j which satisfy the threshold
    j = len(weighted_p_values_flattened)
    while j > 0:
        index = weighted_p_values_flattened_sorted_ind[j-1]
        if np.sum(pis * weighted_p_values_flattened[index], axis=None) / j <= alpha:
            break
        else:
            j -= 1
    # print('j equals %d'%j)
    # output
    output = np.zeros(len(weighted_p_values_flattened))
    if j > 0:
        signal_ind = weighted_p_values_flattened_sorted_ind[:j]  # extract the ind of the signal
        output[signal_ind] = 1
    output = np.reshape(output, p_values.shape)
    return output


def sparsity_estimation_via_screening(p_values, locations, tau, h, kernel='gaussian'):
    """estimate the sparsity level pis

    Args:
        p_values (array_like): original p_values at each location.
        locations (array_like): the spatial location information.
        tau (float): threshold to apply screening
        h (float): bandwidth parameter when doing kernel density estimation
        kernel (str): The form of these kernels is as follows: 'gaussian', 'tophat'



    Returns:
        array_like: sparsity estimation at each location

    """
    n_feature = locations.shape[-1]  # the last dimension of locations is the feature used to calculate the distance
    locations_flattened = locations.reshape([-1, n_feature])

    kd_tree = sklearn.neighbors.KDTree(locations_flattened)
    sum_vs_flattened = kd_tree.kernel_density(locations_flattened, h=h, kernel=kernel)

    p_values_flattened = p_values.flatten()
    locations_flattened_with_p_values_greater_than_tau = locations_flattened[p_values_flattened>tau, :]  # filter those locations with p_values greater than tau
    kd_tree_with_p_values_greater_than_tau = sklearn.neighbors.KDTree(locations_flattened_with_p_values_greater_than_tau)
    sum_vs_flattened_with_p_values_greater_than_tau = kd_tree_with_p_values_greater_than_tau.kernel_density(locations_flattened, h=h, kernel=kernel)

    pis_flattened = 1 - sum_vs_flattened_with_p_values_greater_than_tau / ((1-tau)*sum_vs_flattened)
    # print('the estimated maximum pis is %.4f'%max(pis_flattened))
    pis = pis_flattened.reshape(p_values.shape)

    # stablize the result
    v = 1e-8
    pis[pis>1-v] = 1-v
    pis[pis<v] = v

    return pis


def laws_procedure_2D_grid(p_values, tau=0.1, h=5, alpha=0.05):
    """Laws procedure on a grid

    Args:
        p_values (2d numpy array): original p_values at each location.
        tau (float): threshold to apply screening
        h (float): bandwidth parameter when doing kernel density estimation
        alpha (float): threshold to adjust multiplicity.

    Returns:
        array_like: a binary array with the same dimension as the p_values. The position where a signal is detected is equal to 1.
    """

    # sparsity estimation
    ROW = p_values.shape[0]
    COL = p_values.shape[1]
    x = np.repeat(np.arange(ROW), COL)
    y = np.tile(np.arange(ROW), COL)
    locations = np.stack([x, y]).T.reshape([ROW, COL, 2])

    pis_est = sparsity_estimation_via_screening(p_values, locations, tau, h)

    # laws procedure
    weights = pis_est / (1 - pis_est)
    output = laws_procedure(p_values, pis_est, weights, alpha)
    return output


def locationwise_time_series_anomaly_detection(data, alpha=0.05):
    """Perform separate time-series outlier test for each location

    Args:
        data (3d numpy array): the first two dimensions are space coordinates, the last dimension is time
        alpha (float): unadjusted level of false alarm rate

    Returns:
        studentized residuals (3d array), unadjusted p values (3d array), bonferroni corrected p values (3d array).

    """
    studentized_resid = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
    unadj_pvalue = np.ones([data.shape[0], data.shape[1], data.shape[2]])

    for i in range(data.shape[0]):

        # if i > 0 and i % 10 == 0:
        #     print(i)

        for j in range(data.shape[1]):
            df = pd.DataFrame({'Y': data[i, j, 1:], 'X': data[i, j, :-1], 't': np.arange(1, data.shape[2])})
            fit = statsmodels.formula.api.ols('Y~X+t', data=df).fit()
            outlier = fit.outlier_test()
            # print(outlier)
            # starting from t=2
            studentized_resid[i, j, 1:] = outlier['student_resid']
            unadj_pvalue[i, j, 1:] = outlier['unadj_p']

            unadj_pvalue[studentized_resid<0] = unadj_pvalue[studentized_resid<0] / 2  # one-sided p-value
            unadj_pvalue[studentized_resid>=0] = 1 - unadj_pvalue[studentized_resid>=0] / 2  # one-sided p-value

    return studentized_resid, unadj_pvalue



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
    alpha = 0.1

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

    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            ax[i, j].imshow(data[..., ind], cmap='gray_r')
            ax[i, j].axis('off')

    plt.show()
    fig.savefig('figure/raw_data.png')

    studentized_resid, unadj_pvalue = locationwise_time_series_anomaly_detection(data)


    fig, ax = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            temp = unadj_pvalue[..., ind]
            output = laws_procedure_2D_grid(temp, alpha=alpha)
            ax[i, j].imshow(output, cmap='gray_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()
    fig.savefig('figure/laws_{alpha}.png')

    # without Bonferroni correction
    res = np.zeros(unadj_pvalue.shape)
    res[unadj_pvalue < alpha] = 1
    fig, ax = plt.subplots(4, 5)

    for i in range(4):
        for j in range(5):
            ind = i * 5 + j
            ax[i, j].imshow(res[..., ind], cmap='gray_r')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.show()
    fig.savefig('figure/no_laws_{alpha}.png')

