import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.ops.similarities import gaussian_kernel
import numpy as np
from scipy.spatial.distance import cdist
import os
from scipy.special import kv, gamma
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

# num = 5022
# num = 5000
# num = 5016
num = 5004


# Load the dataset
file_path = 'Female/TagData/LowTag' + str(num) + '.csv'
deer_data = pd.read_csv(file_path)

# load the covariate .tif file
covariate_file_path = 'Female/NLCDClip/LowTag' + str(num) + 'NLCDclip.tif'
covariate_file = rasterio.open(covariate_file_path)

row, col = covariate_file.index(deer_data['X'], deer_data['Y'])
# Assuming row and col are lists of the same length
values = []
roi = covariate_file.read(1)
for r, c in zip(row, col):
    if r < roi.shape[0] and c < roi.shape[1]:
        values.append(covariate_file.read(1)[r, c])
    else:
        values.append(None)

deer_data['covariate'] = values

start_time, end_time = deer_data['jul'].min(), deer_data['jul'].max()
time_interval = 0.16
tolerance = 0.08

T_values = np.arange(start_time, end_time, time_interval)
df = pd.DataFrame(T_values, columns=['T'])


# Function to find nearest row within tolerance
def find_nearest_row_within_tolerance(value, tolerance, dataframe, column_name):
    nearest_idx = (dataframe[column_name] - value).abs().argsort()[:1]
    nearest_value = dataframe[column_name].iloc[nearest_idx].values[0]
    if abs(nearest_value - value) <= tolerance:
        return dataframe.iloc[nearest_idx]
    return pd.DataFrame(columns=dataframe.columns)


# Initialize a list to store dictionaries
data_list = []

# Merge data
for t_value in df['T']:
    matched_row = find_nearest_row_within_tolerance(t_value, tolerance, deer_data, 'jul')
    if not matched_row.empty:
        row_data = {'T': t_value, **matched_row.iloc[0].to_dict()}
    else:
        row_data = {'T': t_value, **{col: np.nan for col in deer_data.columns}}
    data_list.append(row_data)

# Create DataFrame from list of dictionaries
df = pd.DataFrame(data_list)

y = df.loc[:, ['X', 'Y']].values

L = y.shape[0]
C = y.shape[1]

y = y.reshape(L, 1, C)

mask = np.ones_like(y)
mask[np.isnan(y)] = 0
mask = mask.astype(int)

# impute missing values with 0
y[np.isnan(y)] = 0

p_missing = 0.2
rng = np.random.RandomState(42)
time_points_to_eval = rng.choice(L, int(p_missing * L), replace=False)
eval_mask = np.zeros_like(y)
eval_mask[time_points_to_eval, ...] = 1
eval_mask = eval_mask.astype(int)
eval_mask = eval_mask & mask

training_mask = mask & (1 - eval_mask)
y_true = y.copy()
y[training_mask == 0] = np.nan

# impute missing values in y using interpolation
for c in range(C):
    y[:, 0, c] = pd.Series(y[:, 0, c]).interpolate(method='linear', limit_direction='both').values
    # y[:, 0, c] = pd.Series(y[:, 0, c]).interpolate(method='bfill').values


y[np.isnan(y)] = 0

# mae between y and y_true on eval_mask
mae = np.sum(np.abs(y - y_true) * eval_mask) / np.sum(eval_mask)
print(mae)


