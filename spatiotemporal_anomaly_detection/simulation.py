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


def generate_ar_series(n_steps, ar_params=[0.5, -0.2], sigma=1):
    """
    Generate an autoregressive time series of order p.

    Parameters:
        n_steps (int): Number of steps in the time series.
        ar_params (list or np.array): Coefficients of the AR model.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.array: Generated autoregressive time series.
    """
    # randomly initialize the ar_params
    rng = np.random.default_rng(42)
    while True:
        # ar_params = np.random.uniform(-1, 1, 2)
        ar_params = rng.uniform(-1, 1, 2)
        if abs(ar_params[1]) < 1 and abs(ar_params[0] + ar_params[1]) < 1 and abs(ar_params[0] - ar_params[1]) < 1:
            break


    p = len(ar_params)  # Order of the autoregressive model
    # Initialize the time series with zeros
    series = np.zeros(n_steps)
    # Initialize the first `p` values
    series[:p] = np.random.normal(scale=sigma, size=p)

    # Generate the time series
    for i in range(p, n_steps):
        # The next value is a linear combination of the previous `p` values plus noise
        series[i] = np.dot(ar_params, series[i - p:i][::-1]) + np.random.normal(scale=sigma)

    return series




def generate_noise(n_steps, sigma=1):
    # randomly initialize the mu from the standard normal distribution
    rng = np.random.default_rng(42)
    # mu = np.random.normal(0, 1)
    mu = rng.normal(0, 1)


    return np.random.normal(mu, sigma, n_steps)


def generate_seasonal_ts(n_steps, trend_params=(0.1, 5), seasonality_params=[(3, 3), (1.5, 6)], sigma=1):
    """
    Generate a time series with trend, seasonality, and noise.

    Parameters:
        n_steps (int): Number of steps in the time series.
        trend_params (tuple): Parameters for the trend component (slope, intercept).
        seasonality_params (list of tuples): Each tuple contains amplitude and frequency for a sinusoidal component.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.array: Generated time series with seasonality and trend.
    """
    # randomly initialize the trend_params and seasonality_params
    rng = np.random.default_rng(42)
    # trend_params = (np.random.uniform(-1, 1), np.random.uniform(0, 1))
    trend_params = (rng.uniform(-1, 1), rng.uniform(0, 1))
    # seasonality_params[0] = [np.random.uniform(1, 3), np.random.uniform(3, 6)]
    # seasonality_params[1] = [np.random.uniform(1, 3), np.random.uniform(3, 6)]
    seasonality_params[0] = [rng.uniform(1, 3), rng.uniform(3, 6)]
    seasonality_params[1] = [rng.uniform(1, 3), rng.uniform(3, 6)]





    slope, intercept = trend_params
    t = np.arange(n_steps)
    trend = slope * t + intercept  # Linear trend

    seasonality = np.zeros(n_steps)
    for amplitude, frequency in seasonality_params:
        seasonality += amplitude * np.sin(2 * np.pi * frequency * t / n_steps)

    noise = np.random.normal(scale=sigma, size=n_steps)  # Gaussian noise

    return trend + seasonality + noise


def generate_spatiotemporal_data(n_locations, n_steps, func, **kwargs):
    """
    Generate a spatiotemporal dataset by applying a given function to each location.

    Parameters:
        n_locations (int): Number of locations.
        n_steps (int): Number of steps in the time series.
        func (callable): Function that generates a time series.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        np.array: Spatiotemporal dataset with shape (n_locations, n_steps).
    """
    return np.stack([func(n_steps, **kwargs) for _ in range(n_locations)])


def add_point_anomalies(data, dist, affected_time_proportion=0.2, affected_location_proportion=0.1, shock_magnitude=3):
    """
    Introduces point anomalies into a spatiotemporal dataset.

    Parameters:
    - data (np.array): A 2D numpy array with shape (n_locations, n_steps) representing the original dataset.
    - dist (np.array): A 2D numpy array with shape (n_locations, n_locations), capturing the distances between locations.
    - affected_time_proportion (float, optional): The fraction of total time points to be affected by anomalies. Default is 0.2.
    - affected_location_proportion (float, optional): The fraction of locations that will be affected in each anomalous time point. Default is 0.2.
    - shock_magnitude (int, optional): The intensity of the anomaly impact, which can be positive or negative. Default is 10.

    Returns:
    - updated_data (np.array): The modified data array including the anomalies.
    - anomalies (list): A list of tuples, with each tuple containing a time index and the indices of locations where anomalies were added at that time.
    - location_to_times (dict): A dictionary mapping each location index to a list of time indices where anomalies occurred.

    The function operates in several steps:
    1. Randomly select a subset of time points to introduce anomalies.
    2. For each selected time point, randomly choose a subset of locations.
    3. Apply a shock, randomly determined as either positive or negative, to the chosen locations at the chosen times.
    """

    n_locations, n_steps = data.shape
    anomalous_data = data.copy()
    n_anomalous_times = int(affected_time_proportion * n_steps)
    n_anomalous_locations = int(affected_location_proportion * n_locations)
    anomalies = np.zeros((n_locations, n_steps))

    # for each location, find the index of the nearest location
    sorted_locations = np.argsort(dist, axis=1)[:, 1:5]

    # Step 1: Select a subset of time points randomly
    anomalous_time_indices = np.random.choice(n_steps, n_anomalous_times, replace=False)


    for t in anomalous_time_indices:
        # Step 2: Select 3 initial locations randomly
        location_list = np.random.choice(n_locations, size=3)

        while len(location_list) < n_anomalous_locations:
            # randomly choose a location from the location_list
            cur_loc = np.random.choice(location_list)
            # select one of the nearest locations
            nearest_location = np.random.choice(sorted_locations[cur_loc])
            if nearest_location not in location_list:
                location_list = np.append(location_list, nearest_location)


        # Step 3: Add shock
        shock = np.random.choice([-1, 1]) * shock_magnitude  # Random sign for the shock
        anomalous_data[location_list, t] += shock

        # Collect anomaly details
        anomalies[location_list, t] = 1

    return anomalous_data, anomalies


def add_collective_anomalies(data, dist, affected_time_proportion=0.05, affected_location_proportion=0.1,
                             shock_magnitude=3):
    """
    Integrates collective anomalies into a spatiotemporal dataset, where the anomalies are temporally and spatially proximate, spanning contiguous time intervals.

    Parameters:
    data : np.array
        A 2D numpy array with dimensions (n_locations, n_steps), representing the dataset.
    dist : np.array
        A 2D numpy array with dimensions (n_locations, n_locations), detailing the distances between each pair of locations.
    affected_time_proportion : float, optional
        The fraction of total time steps affected by anomalies (default is 0.05).
    affected_location_proportion : float, optional
        The fraction of locations affected at each anomalous time step (default is 0.2).
    shock_magnitude : int, optional
        The intensity of the anomaly impact, which can be either positive or negative (default is 10).

    Returns:
    updated_data : np.array
        The dataset updated with anomalies included.
    anomalies : list
        A list of tuples, each containing a time index and the corresponding indices of affected locations.
    location_to_times : dict
        A dictionary mapping each location index to a list of time indices where anomalies occurred.

    Examples:
    The function modifies the input data array by introducing anomalies at specific locations and times, based on the specified proportions and shock magnitude. The anomalies are introduced in a contiguous block of time steps, starting from a randomly chosen time index, and at locations closest to a randomly selected starting location.
    """
    n_locations, n_steps = data.shape
    anomalous_data = data.copy()
    n_anomalous_times = int(affected_time_proportion * n_steps)
    n_anomalous_locations = int(affected_location_proportion * n_locations)
    anomalies = np.zeros((n_locations, n_steps))
    # for each location, find the index of the nearest location
    sorted_locations = np.argsort(dist, axis=1)[:, 1:5]

    # choose block size from 3 to 5
    block_size = np.random.randint(3, 6)
    # divide n_steps by block_size
    n_blocks = n_steps // block_size
    # how many blocks to choose
    n_blocks_to_choose = n_anomalous_times // block_size
    # randomly choose n_blocks_to_choose blocks
    block_indices = np.random.choice(n_blocks, n_blocks_to_choose, replace=False)
    for i in block_indices:
        contiguous_time_indices = np.arange(i * block_size, (i + 1) * block_size)
        # print(contiguous_time_indices)

        # Step 2: Select 3 initial locations randomly
        location_list = np.random.choice(n_locations, size=3)

        while len(location_list) < n_anomalous_locations:
            # randomly choose a location from the location_list
            cur_loc = np.random.choice(location_list)
            # select one of the nearest locations
            nearest_location = np.random.choice(sorted_locations[cur_loc])
            if nearest_location not in location_list:
                location_list = np.append(location_list, nearest_location)

        # Randomly determine the sign of the shock (positive or negative)
        shock = np.random.choice([-1, 1]) * shock_magnitude

        for t in contiguous_time_indices:
            anomalous_data[location_list, t] += shock
            # Collect anomaly details
            anomalies[location_list, t] = 1

    return anomalous_data, anomalies


def time_series_outlier_test(anomalous_data):
    n_locations = anomalous_data.shape[0]
    n_steps = anomalous_data.shape[1]

    studentized_resid = np.zeros([n_locations, n_steps])
    unadj_pvalue = np.ones([n_locations, n_steps])
    bonf_pvalue = np.ones([n_locations, n_steps])
    for i in range(n_locations):
        df = pd.DataFrame({'Y': anomalous_data[i, 1:], 'X': anomalous_data[i, :-1], 't': np.arange(1, n_steps)})
        fit = ols('Y~X+t', data=df).fit()
        outlier = fit.outlier_test()
        # print(outlier)
        # starting from t=2
        studentized_resid[i, 1:] = outlier['student_resid']
        unadj_pvalue[i, 1:] = outlier['unadj_p']
        bonf_pvalue[i, 1:] = outlier['unadj_p'] * n_steps  # bonferroni correction

    return unadj_pvalue


def time_series_anomaly_detection(anomalous_data, horizon=1, input_size=10):
    """
    Detects anomalies in a time series dataset using the DLinear model from NeuralForecast.

    Args:
    anomalous_data (pd.DataFrame): Input DataFrame containing the time series data.
    horizon (int): The forecast horizon and input size for the DLinear model.

    Returns:
    np.ndarray: An array of p-values indicating the anomaly likelihood for each time point.
    """
    n_locations, n_steps = anomalous_data.shape

    # Create a DataFrame
    df_anomalous = pd.DataFrame(anomalous_data)

    # Melt the DataFrame to long format
    df_anomalous = df_anomalous.reset_index().melt(id_vars='index', var_name='time', value_name='y')

    # Rename columns
    df_anomalous.columns = ['unique_id', 'ds', 'y']

    # Convert the 'ds' column to integers
    df_anomalous['ds'] = df_anomalous['ds'].astype(int)

    nf = NeuralForecast(
        models=[DLinear(input_size=input_size, h=horizon, max_steps=10000)],
        freq=1
    )
    val_size = int(0.2 * n_steps)

    nf.fit(df=df_anomalous, val_size=val_size)
    df_pred = nf.predict_insample(step_size=horizon)

    df_pred.loc[df_pred['ds'] < horizon, 'DLinear'] = df_pred.loc[df_pred['ds'] < horizon, 'y']
    df_pred.loc[:, 'reconstruction_error'] = np.abs(df_pred.loc[:, 'y'] - df_pred.loc[:, 'DLinear'])

    # # Empirical p-values
    # # calculate the empirical p-values of the reconstruction error
    # df_pred.loc[:, 'p_value'] = 1 - df_pred.loc[:, 'reconstruction_error'].rank(pct=True)

    # use IQR to detect outliers
    reconstruction_error = df_pred.loc[df_pred['ds'] >= horizon, 'reconstruction_error']
    q1 = reconstruction_error.quantile(0.25)
    q3 = reconstruction_error.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # assume the reconstruction error follows a normal distribution for normal points
    mean = np.mean(reconstruction_error[reconstruction_error < upper_bound])
    std = np.std(reconstruction_error[reconstruction_error < upper_bound])

    # p-value is the 1-F(x), where F(x) is the CDF of the normal distribution with mean and std
    df_pred.loc[:, 'p_value'] = 1 - norm.cdf(df_pred.loc[:, 'reconstruction_error'], loc=mean, scale=std)

    df_pred = df_pred.reset_index(drop=False)

    # Sort the DataFrame by 'unique_id' and 'ds' (if 'ds' is not sorted)
    df_pred = df_pred.sort_values(by=['unique_id', 'ds'])
    # Pivot the DataFrame
    pivot_df = df_pred.pivot(index='unique_id', columns='ds', values='p_value')

    # Convert the pivoted DataFrame to a NumPy array
    p_values = pivot_df.values

    return p_values


def laws_procedure(p_values, distance_matrix, alpha=0.05, tau=0.1, h=5, kernel='gaussian'):
    """
    Implements the LAWS procedure for False Discovery Rate (FDR) control using spatial information
    and adaptive thresholding.

    This function estimates sparsity levels based on a distance matrix and then adjusts the p-values
    using these estimates to control the FDR. The method accounts for varying sparsity levels and
    adaptive weights across multiple locations.

    Args:
        p_values (array_like): An array of original p-values at each location.
        distance_matrix (array_like): A matrix representing distances between locations, used to estimate sparsity levels.
        alpha (float or list, optional): The threshold level(s) for controlling FDR. Default is 0.05.
        tau (float, optional): A parameter used in the estimation of sparsity levels. Default is 0.1.
        h (int, optional): The bandwidth parameter used in the kernel density estimation. Default is 5.
        kernel (str, optional): The type of kernel used in the density estimation. Default is 'gaussian'.

    Returns:
        array_like or list of arrays: A binary array or a list of binary arrays of the same dimension as p_values, where positions with detected signals
                                     are marked as 1, and others as 0.

    Example:
        >>> p_values = np.random.rand(100)
        >>> distance_matrix = np.random.rand(100, 100)
        >>> results = laws_procedure(p_values, distance_matrix, alpha=[0.05, 0.1])
        >>> for result in results:
        ...     print(result)
    """

    # calculate the sparsity level
    pis = sparsity_estimation_via_distance_matrix(p_values, distance_matrix, tau=tau, h=h, kernel=kernel)
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


def sparsity_estimation_via_distance_matrix(p_values, dist=None, tau=0.1, h=5, kernel='gaussian'):

    n_locations = len(p_values)
    grid_size = int(np.sqrt(n_locations))
    locations = np.array([(i // grid_size, i % grid_size) for i in range(grid_size ** 2)])

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



def spatiotemporal_anomaly_detection(anomalous_data, dist, ts='NN', laws=True, **kwargs):
    """
    Detects spatiotemporal anomalies in a dataset. This function first apply time series anomaly detection to obtain p-values. It then applies
    the LAWS procedure with a list of defined alpha thresholds to these p-values to identify anomalies.

    Parameters:
        anomalous_data (numpy.ndarray): A 2D array with shape (n_locations, n_steps) containing
            the data where anomalies are to be detected. Each row represents a location and each
            column a time step.
        dist (numpy.ndarray): A 2D array with shape (n_locations, n_locations) containing the distances
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
        p_values = time_series_anomaly_detection(anomalous_data, **kwargs)
    elif ts == 'outlier_test':
        p_values = time_series_outlier_test(anomalous_data)
    else:
        raise ValueError("Invalid time series anomaly detection method. Choose 'NN' or 'outlier_test'.")

    # alpha_list = np.linspace(0.01, 0.1, 100)
    alpha_list = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, float('inf')]
    res = np.zeros((len(alpha_list), n_locations, n_steps))

    if laws:
        for i in range(n_steps):
            res[:, :, i] = laws_procedure(p_values[:, i], dist, alpha=alpha_list)
    else:
        for i in range(len(alpha_list)):
            res[i] = p_values < alpha_list[i]

    return res


def calculate_tpr_fpr(y_true, y_pred):
    """
    Calculate the True Positive Rate (TPR) and False Positive Rate (FPR).

    Parameters:
        y_true (numpy.array): The ground truth binary labels (1s and 0s).
        y_pred (numpy.array): The predicted binary labels (1s and 0s).

    Returns:
        tuple: A tuple containing the TPR and FPR.
    """
    # Convert to numpy arrays for safety
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # True Positives (TP): The predictions and labels are both 1.
    TP = np.sum((y_pred == 1) & (y_true == 1))

    # False Negatives (FN): The predictions are 0 but the labels are 1.
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # False Positives (FP): The predictions are 1 but the labels are 0.
    FP = np.sum((y_pred == 1) & (y_true == 0))

    # True Negatives (TN): The predictions and labels are both 0.
    TN = np.sum((y_pred == 0) & (y_true == 0))

    # Calculate TPR and FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (TN + FP) if (TN + FP) > 0 else 0

    return TPR, FPR


def plot_roc_and_calculate_auc(y_true, y_pred_at_different_thresholds):
    """
    Plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC)
    for a set of predictions at different thresholds.

    This function iterates over a list of threshold-specific predicted values, calculating the true positive rate
    (TPR) and false positive rate (FPR) for each threshold. It then uses these values to plot the ROC curve and
    calculate the AUC.

    Parameters:
    - y_true (list of int): The ground truth binary labels, where each element is 0 or 1.
    - y_pred_at_different_thresholds (list of list of float): A list of lists where each sublist contains predicted
      probabilities for each threshold, corresponding to the prediction of the positive class.

    Returns:
    - roc_auc (float): The calculated AUC of the ROC curve, representing the model's ability to discriminate between
      the positive and negative classes.

    Notes:
    - The function assumes that 'calculate_tpr_fpr' is defined elsewhere to compute TPR and FPR.
    - It also assumes the use of 'auc' from an appropriate library for AUC calculation, and 'plt' from matplotlib
      for plotting.

    Raises:
    - ValueError: If 'y_true' and 'y_pred_at_different_thresholds' have different lengths.
    """

    tpr_list = []
    fpr_list = []

    # Validate input lengths
    if len(y_true) != len(y_pred_at_different_thresholds[0]):
        raise ValueError("Mismatch in number of elements between 'y_true' and predictions for each threshold.")

    for predictions in y_pred_at_different_thresholds:
        tpr, fpr = calculate_tpr_fpr(y_true, predictions)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate the AUC
    roc_auc = auc(fpr_list, tpr_list)

    # # Plot the ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    return roc_auc


def run_simulation(config):
    df_simulation_result = pd.DataFrame(
        columns=['type of time series', 'type of anomalies', 'shock magnitude', 'temporal method', 'spatial method', 'auc'])
    type_of_time_series = config['type_of_time_series']
    type_of_anomalies = config['type_of_anomalies']
    temporal_methods = config['temporal_methods']
    spatial_methods = config['spatial_methods']
    n_locations = config['n_locations']
    grid_size = config['grid_size']
    n_steps = config['n_steps']
    affected_time_proportion = config['affected_time_proportion']
    affected_location_proportion = config['affected_location_proportion']
    shock_magnitude = config['shock_magnitude']

    for time_series_name in type_of_time_series:
        if time_series_name == 'ar':
            time_series_generator = generate_ar_series
        elif time_series_name == 'trend_seasonal':
            time_series_generator = generate_seasonal_ts
        elif time_series_name == 'iid_noise':
            time_series_generator = generate_noise
        data = generate_spatiotemporal_data(n_locations, n_steps, time_series_generator)
        coordinates = np.array([(i // grid_size, i % grid_size) for i in range(grid_size ** 2)])
        dist = cdist(coordinates, coordinates, metric='euclidean')

        for shock in shock_magnitude:
            for anomaly_type in type_of_anomalies:
                if anomaly_type == 'point':
                    anomalous_data, anomalies = add_point_anomalies(data, dist,
                                                                    affected_time_proportion=affected_time_proportion,
                                                                    affected_location_proportion=affected_location_proportion,
                                                                    shock_magnitude=shock)
                else:
                    anomalous_data, anomalies = add_collective_anomalies(data, dist,
                                                                         affected_time_proportion=affected_time_proportion,
                                                                         affected_location_proportion=affected_location_proportion,
                                                                         shock_magnitude=shock)

                for temporal_method in temporal_methods:
                    for spatial_method in spatial_methods:
                        if spatial_method == 'laws':
                            res = spatiotemporal_anomaly_detection(anomalous_data, dist, temporal_method, laws=True)
                        else:
                            res = spatiotemporal_anomaly_detection(anomalous_data, dist, temporal_method, laws=False)
                        auc = plot_roc_and_calculate_auc(anomalies, res)
                        df_simulation_result.loc[len(df_simulation_result)] = [time_series_name, anomaly_type, shock,
                                                                               temporal_method, spatial_method, auc]

    # save the result
    df_simulation_result.to_csv('./results/simulation_result.csv', index=False)

    return df_simulation_result



if __name__ == '__main__':
    config = {
        'type_of_time_series': ['trend_seasonal', 'iid_noise', 'ar'],
        'type_of_anomalies': ['point', 'collective'],
        'temporal_methods': ['NN', 'outlier_test'],
        'spatial_methods': ['laws', 'no laws'],
        'n_locations': 400,
        'grid_size': 20,
        'n_steps': 20,
        'affected_time_proportion': 0.2,
        'affected_location_proportion': 0.2,
        'shock_magnitude': [3, 2, 1]
    }

    # config = {
    #     'type_of_time_series': ['iid_noise'],
    #     'type_of_anomalies': ['point'],
    #     'temporal_methods': ['outlier_test'],
    #     'spatial_methods': ['laws'],
    #     'n_locations': 10000,
    #     'grid_size': 100,
    #     'n_steps': 20,
    #     'affected_time_proportion': 0.1,
    #     'affected_location_proportion': 0.1,
    #     'shock_magnitude': [1]
    # }

    run_simulation(config)

