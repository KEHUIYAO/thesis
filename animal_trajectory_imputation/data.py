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

from tsl.data.datamodule.splitters import Splitter
from datetime import datetime, timedelta


current_dir = os.path.dirname(os.path.abspath(__file__))


class AnimalMovement():
    def __init__(self, mode='train', deer_id=0, p_missing=0.2):
        if mode == 'train':
            deer_id_list = sorted([int(f.split('.')[0][-4:]) for f in os.listdir('Female/TagData') if f.endswith('.csv')])
            # randomly select 80& of the deer ids as training data
            rng = np.random.RandomState(42)
            rng.shuffle(deer_id_list)
            deer_id_list = deer_id_list[:int(0.8 * len(deer_id_list))]
        else:
            deer_id_list = [deer_id]

        y_list = []
        X_list = []
        for deer_id in deer_id_list:
            num = deer_id

            try:
                df = self.load_data(num)
            except:
                continue

            if mode == 'imputation':
                df = self.data_augmentation(df)


            y = df.loc[:, ['X', 'Y']].values
            y = y.astype(float)


            # covariates
            X = df.loc[:, ['jul', 'month', 'day', 'hour', 'covariate']].values
            # replace missing values with 0
            X[np.isnan(X)] = 0

            cutoff = int(y.shape[0] / 72) * 72
            y = y[:cutoff]
            X = X[:cutoff]


            y_list.append(y)
            X_list.append(X)
            # # append 100 rows of np.nan to the list
            # y_list.append(np.full((100, 2), np.nan))
            # X_list.append(np.full((100, 5), 0))

        y = np.concatenate(y_list, axis=0)
        X = np.concatenate(X_list, axis=0)

        # reshape y
        L = y.shape[0]
        C = y.shape[1]
        y = y.reshape(L, 1, C)

        # one-hot encoding for covariates
        covariates = X[:, 4]
        # print unique values of covariates
        print(np.unique(covariates))

        unique_covariates = np.array([0, 11, 21, 22, 23, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95, 128])
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(unique_covariates.reshape(-1, 1))
        covariates = encoder.transform(covariates.reshape(-1, 1))

        # normalize month, day, and hour to [0, 1]
        jul = X[:, 0]
        month = X[:, 1]
        day = X[:, 2]
        hour = X[:, 3]

        # concatenate month, day, hour and covariates
        X = np.concatenate([jul.reshape(-1, 1), month.reshape(-1, 1), day.reshape(-1, 1), hour.reshape(-1, 1), covariates], axis=1)

        X = X.reshape(L, 1, X.shape[1])


        # randomly set 20% of data to be missing as test data
        mask = np.ones_like(y)
        mask[np.isnan(y)] = 0
        mask = mask.astype(int)

        if mode == 'imputation':
            p_missing = 0.0
        rng = np.random.RandomState(42)
        time_points_to_eval = rng.choice(L, int(p_missing * L), replace=False)
        eval_mask = np.zeros_like(y)
        eval_mask[time_points_to_eval, ...] = 1
        eval_mask = eval_mask.astype(int)
        eval_mask = eval_mask & mask
        y[np.isnan(y)] = 0
        self.eval_mask = eval_mask
        self.training_mask = mask & (1 - eval_mask)
        self.y = y
        self.attributes = {}
        space_coords, time_coords = np.meshgrid(np.arange(1), np.arange(L))
        st_coords = np.stack([space_coords, time_coords], axis=-1)
        self.attributes['st_coords'] = st_coords

        X[time_points_to_eval, 4:] = 0

        self.attributes['covariates'] = X

    def load_data(self, num):

        # if the processed file is already existed, load it
        if os.path.exists(f'./Female/Processed/{num}.csv'):
            return pd.read_csv('Female/Processed/' + str(num) + '.csv')

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



        base_date = datetime(2017, 1, 1, 0, 0, 0)
        deer_data['date'] = [base_date + timedelta(days=x) for x in deer_data['jul']]
        deer_data['month'] = [x.month for x in deer_data['date']]
        deer_data['day'] = [x.day for x in deer_data['date']]
        deer_data['hour'] = [x.hour for x in deer_data['date']]

        fig, axs = plt.subplots(2, figsize=(10, 5))

        # x axis is the year-month-day, y axis is the X or Y coordinates
        axs[0].plot(deer_data['jul'], deer_data['X'], 'o', markersize=1)
        axs[1].plot(deer_data['jul'], deer_data['Y'], 'o', markersize=1)

        # don't make x axis label overlap
        plt.tight_layout()


        # create a folder called result to save the figure
        if not os.path.exists(f'results/{num}'):
            os.makedirs(f'results/{num}')

        # save fig to file, file name is the deer id
        fig.savefig(f'results/{num}/original.png')

        # save the dataframe to a csv file
        # if the folder is not existed, create it
        if not os.path.exists(f'Female/Processed'):
            os.makedirs(f'Female/Processed')

        deer_data.to_csv(f'Female/Processed/{num}.csv', index=False)

        return deer_data


    def data_augmentation(self, deer_data, time_interval=0.16, tolerance=0.08):
        start_time, end_time = deer_data['jul'].min(), deer_data['jul'].max()
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
        df_matched = pd.DataFrame(data_list)

        base_date = datetime(2017, 1, 1, 0, 0, 0)
        df_matched['date'] = [base_date + timedelta(days=x) for x in df_matched['T']]
        df_matched['month'] = [x.month for x in df_matched['date']]
        df_matched['day'] = [x.day for x in df_matched['date']]
        df_matched['hour'] = [x.hour for x in df_matched['date']]

        df_matched['jul'] = df_matched['T']
        df_matched = df_matched.drop(columns=['T'])

        return df_matched


    def get_splitter(self, val_len, test_len):
        return AnimalMovementSplitter(val_len, test_len)


class AnimalMovementSplitter(Splitter):

    def __init__(self, val_len: int = None, test_len: int = None):
        super().__init__()
        self._val_len = val_len
        self._test_len = test_len

    def fit(self, dataset):
        idx = np.arange(len(dataset))
        val_len, test_len = self._val_len, self._test_len
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))

        # randomly split idx into train, val, test
        np.random.shuffle(idx)
        val_start = len(idx) - val_len - test_len
        test_start = len(idx) - test_len

        self.set_indices(idx[:val_start],
                         idx[val_start:test_start],
                         idx[test_start:])

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--val-len', type=float or int, default=0.2)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser


if __name__ == '__main__':
    dataset = AnimalMovement()

