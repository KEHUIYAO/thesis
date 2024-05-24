import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import auc
from anomaly_detection import spatiotemporal_anomaly_detection


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
    # rng = np.random.default_rng(42)
    while True:
        ar_params = np.random.uniform(-1, 1, 2)
       # ar_params = rng.uniform(-1, 1, 2)
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
    mu = np.random.normal(0, 1)
    # mu = rng.normal(0, 1)


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
    # rng = np.random.default_rng(42)
    trend_params = (np.random.uniform(-1, 1), np.random.uniform(0, 1))
    # trend_params = (rng.uniform(-1, 1), rng.uniform(0, 1))
    seasonality_params[0] = [np.random.uniform(1, 3), np.random.uniform(3, 6)]
    seasonality_params[1] = [np.random.uniform(1, 3), np.random.uniform(
    3, 6)]
    # seasonality_params[0] = [rng.uniform(1, 3), rng.uniform(3, 6)]
    # seasonality_params[1] = [rng.uniform(1, 3), rng.uniform(3, 6)]


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
    anomalous_data : np.array
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
    n_blocks_to_choose = max(1, n_anomalous_times // block_size)
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
        locations = np.array([(i // grid_size, i % grid_size) for i in range(grid_size ** 2)])
        dist = cdist(locations, locations, metric='euclidean')

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
                            res = spatiotemporal_anomaly_detection(anomalous_data, locations, temporal_method, laws=True, horizon=config['horizon'], input_size=config['input_size'])
                        else:
                            res = spatiotemporal_anomaly_detection(anomalous_data, locations, temporal_method, laws=False, horizon=config['horizon'], input_size=config['input_size'])
                        auc = plot_roc_and_calculate_auc(anomalies, res)
                        df_simulation_result.loc[len(df_simulation_result)] = [time_series_name, anomaly_type, shock,
                                                                               temporal_method, spatial_method, auc]

    # save the result
    name = config['name']
    df_simulation_result.to_csv(f'./results/{name}.csv', index=False)

    return df_simulation_result



if __name__ == '__main__':
    # config = {
    #     'name': 'simulation_short_time_series',
    #     'type_of_time_series': ['trend_seasonal', 'iid_noise', 'ar'],
    #     'type_of_anomalies': ['point', 'collective'],
    #     'temporal_methods': ['NN', 'outlier_test'],
    #     'spatial_methods': ['laws', 'no laws'],
    #     'n_locations': 400,
    #     'grid_size': 20,
    #     'n_steps': 20,
    #     'affected_time_proportion': 0.1,
    #     'affected_location_proportion': 0.1,
    #     'shock_magnitude': [3, 2, 1],
    #     'horizon': 1,
    #     'input_size': 1
    # }
    #
    # run_simulation(config)
    #
    # config = {
    #     'name': 'simulation_long_time_series',
    #     'type_of_time_series': ['trend_seasonal', 'iid_noise', 'ar'],
    #     'type_of_anomalies': ['point', 'collective'],
    #     'temporal_methods': ['NN', 'outlier_test'],
    #     'spatial_methods': ['laws', 'no_laws'],
    #     'n_locations': 400,
    #     'grid_size': 20,
    #     'n_steps': 500,
    #     'affected_time_proportion': 0.1,
    #     'affected_location_proportion': 0.1,
    #     'shock_magnitude': [3, 2, 1],
    #     'horizon': 1,
    #     'input_size': 10
    # }
    #
    # run_simulation(config)


    config = {
        'name': 'simulation_test',
        'type_of_time_series': ['ar'],
        'type_of_anomalies': ['point'],
        'temporal_methods': ['NN', 'outlier_test'],
        'spatial_methods': ['no_laws'],
        'n_locations': 400,
        'grid_size': 20,
        'n_steps': 100,
        'affected_time_proportion': 0.1,
        'affected_location_proportion': 0.1,
        'shock_magnitude': [3, 2, 1],
        'horizon': 1,
        'input_size': 10
    }

    run_simulation(config)

