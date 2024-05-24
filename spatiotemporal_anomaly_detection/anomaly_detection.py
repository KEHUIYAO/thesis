import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from sklearn.metrics import auc
from statsmodels.formula.api import ols
from scipy.stats import norm
import time
import sklearn.neighbors
import matplotlib.pyplot as plt
from dlinear import Model, LightningModel, STData, Configs
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import scipy.stats

def time_series_outlier_test(anomalous_data, one_sided='None'):
    n_locations = anomalous_data.shape[0]
    n_steps = anomalous_data.shape[1]
    studentized_resid = np.zeros([n_locations, n_steps])
    unadj_pvalue = np.ones([n_locations, n_steps])
    for i in range(n_locations):
        df = pd.DataFrame({'Y': anomalous_data[i, 1:], 'X': anomalous_data[i, :-1], 't': np.arange(1, n_steps)})
        fit = ols('Y~X+t', data=df).fit()
        outlier = fit.outlier_test()
        # print(outlier)
        # starting from t=2
        studentized_resid[i, 1:] = outlier['student_resid']
        if one_sided == 'None':
            unadj_pvalue[i, 1:] = outlier['unadj_p']
        elif one_sided == 'left':
            unadj_pvalue[i, 1:] = scipy.stats.t.cdf(studentized_resid[i, 1:], df=n_steps-5)
        elif one_sided == 'right':
            unadj_pvalue[i, 1:] = 1 - scipy.stats.t.cdf(studentized_resid[i, 1:], df=n_steps-5)

    return unadj_pvalue


# def time_series_anomaly_detection(anomalous_data, horizon=1, input_size=10):
#     """
#     Detects anomalies in a time series dataset using the DLinear model from NeuralForecast.
#
#     Args:
#     anomalous_data (pd.DataFrame): Input DataFrame containing the time series data.
#     horizon (int): The forecast horizon and input size for the DLinear model.
#
#     Returns:
#     np.ndarray: An array of p-values indicating the anomaly likelihood for each time point.
#     """
#     n_locations, n_steps = anomalous_data.shape
#
#     # Create a DataFrame
#     df_anomalous = pd.DataFrame(anomalous_data)
#
#     # Melt the DataFrame to long format
#     df_anomalous = df_anomalous.reset_index().melt(id_vars='index', var_name='time', value_name='y')
#
#     # Rename columns
#     df_anomalous.columns = ['unique_id', 'ds', 'y']
#
#     # Convert the 'ds' column to integers
#     df_anomalous['ds'] = df_anomalous['ds'].astype(int)
#
#     nf = NeuralForecast(
#         models=[DLinear(input_size=input_size, h=horizon, max_steps=10000)],
#         freq=1
#     )
#     val_size = int(0.2 * n_steps)
#
#     nf.fit(df=df_anomalous, val_size=val_size)
#     df_pred = nf.predict_insample(step_size=horizon)
#
#     df_pred.loc[df_pred['ds'] < horizon, 'DLinear'] = df_pred.loc[df_pred['ds'] < horizon, 'y']
#     df_pred.loc[:, 'reconstruction_error'] = np.abs(df_pred.loc[:, 'y'] - df_pred.loc[:, 'DLinear'])
#
#     # # Empirical p-values
#     # # calculate the empirical p-values of the reconstruction error
#     # df_pred.loc[:, 'p_value'] = 1 - df_pred.loc[:, 'reconstruction_error'].rank(pct=True)
#
#     # use IQR to detect outliers
#     reconstruction_error = df_pred.loc[df_pred['ds'] >= horizon, 'reconstruction_error']
#     q1 = reconstruction_error.quantile(0.25)
#     q3 = reconstruction_error.quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#
#     # assume the reconstruction error follows a normal distribution for normal points
#     mean = np.mean(reconstruction_error[reconstruction_error < upper_bound])
#     std = np.std(reconstruction_error[reconstruction_error < upper_bound])
#
#     # p-value is the 1-F(x), where F(x) is the CDF of the normal distribution with mean and std
#     df_pred.loc[:, 'p_value'] = 1 - norm.cdf(df_pred.loc[:, 'reconstruction_error'], loc=mean, scale=std)
#
#     df_pred = df_pred.reset_index(drop=False)
#
#     # Sort the DataFrame by 'unique_id' and 'ds' (if 'ds' is not sorted)
#     df_pred = df_pred.sort_values(by=['unique_id', 'ds'])
#     # Pivot the DataFrame
#     pivot_df = df_pred.pivot(index='unique_id', columns='ds', values='p_value')
#
#     # Convert the pivoted DataFrame to a NumPy array
#     p_values = pivot_df.values
#
#     return p_values


def time_series_anomaly_detection(anomalous_data, horizon=1, input_size=1, one_sided='None'):
    """
    Detects anomalies in a time series dataset using the DLinear model from NeuralForecast.

    Args:
    anomalous_data (pd.DataFrame): Input DataFrame containing the time series data.
    horizon (int): The forecast horizon and input size for the DLinear model.

    Returns:
    np.ndarray: An array of p-values indicating the anomaly likelihood for each time point.
    """
    n_locations, n_steps = anomalous_data.shape

    dataset = STData(anomalous_data, input_len=input_size, pred_len=horizon, stride=horizon)

    training_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    configs = Configs(seq_len=input_size, pred_len=horizon, individual=False, enc_in=n_locations)
    model = Model(configs)
    lightning_model = LightningModel(model)

    trainer = pl.Trainer(max_epochs=500)
    trainer.fit(lightning_model, training_dataloader)

    # Generate predictions
    predictions = trainer.predict(lightning_model, test_dataloader)

    # concatenate predictions along the first dimension, transpose the predictions to [num_channels, pred_len]
    predictions = torch.cat(predictions, dim=0).squeeze().transpose(1, 0).detach().numpy()

    reconstruction_error = predictions - anomalous_data[:, input_size:]

    # # plot the histogram of the reconstruction error
    # plt.hist(reconstruction_error.flatten(), bins=100)
    # plt.show()


    # use IQR to detect outliers
    q1 = np.percentile(reconstruction_error, 25)
    q3 = np.percentile(reconstruction_error, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # assume the reconstruction error follows a normal distribution for normal points
    mean = np.mean(reconstruction_error[(reconstruction_error < upper_bound) & (reconstruction_error > lower_bound)])
    std = np.std(reconstruction_error[(reconstruction_error < upper_bound) & (reconstruction_error > lower_bound)])

    # p-value is the 1-F(x), where F(x) is the CDF of the normal distribution with mean and std
    p_values = np.ones((n_locations, n_steps))
    temp = norm.cdf(reconstruction_error, loc=mean, scale=std)

    if one_sided == 'None':
        p_values[:, input_size:] = np.minimum(1 - temp, temp) * 2
    elif one_sided == 'left':
        p_values[:, input_size:] = temp
    elif one_sided == 'right':
        p_values[:, input_size:] = 1 - temp

    return p_values



def laws_procedure(p_values, locations, alpha=0.05, tau=0.1, h=5, kernel='gaussian'):
    """
    Implements the LAWS procedure for False Discovery Rate (FDR) control using spatial information
    and adaptive thresholding.

    This function estimates sparsity levels based on a distance matrix and then adjusts the p-values
    using these estimates to control the FDR. The method accounts for varying sparsity levels and
    adaptive weights across multiple locations.

    Args:
        p_values (array_like): An array of original p-values at each location.
        locations (array_like): A 2d array of location coordinates.
        alpha (float or list, optional): The threshold level(s) for controlling FDR. Default is 0.05.
        tau (float, optional): A parameter used in the estimation of sparsity levels. Default is 0.1.
        h (int, optional): The bandwidth parameter used in the kernel density estimation. Default is 5.
        kernel (str, optional): The type of kernel used in the density estimation. Default is 'gaussian'.

    Returns:
        array_like or list of arrays: A binary array or a list of binary arrays of the same dimension as p_values, where positions with detected signals
                                     are marked as 1, and others as 0.
    """

    # calculate the sparsity level
    pis = sparsity_estimation_via_distance_matrix(p_values, locations, tau=tau, h=h, kernel=kernel)
    weights = pis / (1 - pis)

    # calculate weighted p-values
    weighted_p_values = p_values / weights

    # order the weighted p-values
    weighted_p_values_sorted_ind = np.argsort(weighted_p_values)

    # if alpha is a single value
    if isinstance(alpha, (int, float)):
        alpha = [alpha]

    output_list = []
    for a in alpha:
        # find the largest j which satisfy the threshold
        j = len(weighted_p_values)
        while j > 0:
            index = weighted_p_values_sorted_ind[j - 1]
            if np.sum(pis * weighted_p_values[index], axis=None) / j <= a:
                break
            else:
                j -= 1

        # output
        output = np.zeros(len(weighted_p_values))
        if j > 0:
            signal_ind = weighted_p_values_sorted_ind[:j]  # extract the ind of the signal
            output[signal_ind] = 1
        output_list.append(output)

    if len(output_list) == 1:
        return output_list[0]

    return output_list


def sparsity_estimation_via_distance_matrix(p_values, locations, tau=0.1, h=5, kernel='gaussian'):
    kd_tree = sklearn.neighbors.KDTree(locations)
    sum_vs = kd_tree.kernel_density(locations, h=h, kernel=kernel)


    locations_with_p_values_greater_than_tau = locations[p_values>tau, :]  # filter those locations with p_values greater than tau
    kd_tree_with_p_values_greater_than_tau = sklearn.neighbors.KDTree(locations_with_p_values_greater_than_tau)
    sum_vs_with_p_values_greater_than_tau = kd_tree_with_p_values_greater_than_tau.kernel_density(locations, h=h, kernel=kernel)

    pis = 1 - sum_vs_with_p_values_greater_than_tau / ((1-tau)*sum_vs)

    # stablize the result
    v = 1e-8
    pis[pis>1-v] = 1-v
    pis[pis<v] = v

    return pis



def spatiotemporal_anomaly_detection(anomalous_data, locations, ts='NN', laws=True,**kwargs):
    """
    Detects spatiotemporal anomalies in a dataset. This function first apply time series anomaly detection to obtain p-values. It then applies
    the LAWS procedure with a list of defined alpha thresholds to these p-values to identify anomalies.

    Parameters:
        anomalous_data (numpy.ndarray): A 2D array with shape (n_locations, n_steps) containing
            the data where anomalies are to be detected. Each row represents a location and each
            column a time step.
        locations (numpy.ndarray): A 2D array with shape (n_locations, 2) containing the spatial coordinates.
        ts (str): The type of time series anomaly detection method to use. Default is 'NN' (neural network).
        laws (bool): A flag indicating whether to apply the LAWS procedure to acknowledge the spatial
            information. Default is True.
        **kwargs: Additional keyword arguments to pass to the time series anomaly detection function.
    Returns:
        numpy.ndarray: A 3D array with shape (num_of_thresholds, n_locations, n_steps) indicating
            the anomaly detection results. Each "layer" in the first dimension corresponds to a
            threshold level applied to the p-values.

    Note:
        The alpha thresholds used are predefined within the function ranging from 0 to 'inf'.
        These thresholds define the sensitivity of the anomaly detection, with lower values
        indicating stricter criteria for anomalies.
    """
    n_locations, n_steps = anomalous_data.shape

    # time series anomaly detection
    if ts == 'NN':
        temp = {key: kwargs[key] for key in ['horizon', 'input_size', 'one_sided'] if key in kwargs}
        p_values = time_series_anomaly_detection(anomalous_data, **temp)
    elif ts == 'outlier_test':
        temp = {key: kwargs[key] for key in ['one_sided'] if key in kwargs}
        p_values = time_series_outlier_test(anomalous_data, **temp)
    else:
        raise ValueError("Invalid time series anomaly detection method. Choose 'NN' or 'outlier_test'.")

    # alpha_list = np.linspace(0.01, 0.1, 100)
    alpha_list = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, float('inf')]
    res = np.zeros((len(alpha_list), n_locations, n_steps))

    if laws:
        for i in range(n_steps):
            res[:, :, i] = laws_procedure(p_values[:, i], locations, alpha=alpha_list)
    else:
        for i in range(len(alpha_list)):
            res[i] = p_values < alpha_list[i]

    return res
