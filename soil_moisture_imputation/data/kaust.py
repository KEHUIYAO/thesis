import pandas as pd
import torch

from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin

from tsl.ops.similarities import gaussian_kernel

from tsl.data.datamodule.splitters import Splitter
import matplotlib.pyplot as plt
import numpy as np


from scipy.spatial.distance import cdist
import os
from sklearn.preprocessing import StandardScaler



current_dir = os.path.dirname(os.path.abspath(__file__))




class Kaust(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self, dataset_index=1):
        self.dataset_index = dataset_index
        df_train_path = '2a_' + str(self.dataset_index) + '_train.csv'
        df_train = pd.read_csv(df_train_path)
        df_train['mask'] = 1
        X_test_path = '2a_' + str(self.dataset_index) + '_test.csv'
        X_test = pd.read_csv(X_test_path)
        y_test_path = '2a-solutions.csv'
        y_test = pd.read_csv(y_test_path).loc[:, ['z' + str(self.dataset_index)]]
        y_test.columns = ['z']

        # concatenate the test data
        df_test = pd.concat([X_test, y_test], axis=1)
        df_test['mask'] = 0

        # concatenate the train and test data
        df = pd.concat([df_train, df_test], axis=0)

        # order by (x,y) and then t
        df = df.sort_values(by=['x', 'y', 't'])

        y = df['z'].values
        mask = df['mask'].values
        space_coords = df[['x', 'y']].drop_duplicates().values
        time_coords = df['t'].drop_duplicates().values

        y_missing = y.copy()
        y_missing[mask == 0] = np.nan

        y_st = y.reshape(len(space_coords), len(time_coords))
        y_st_missing = y_missing.reshape(len(space_coords), len(time_coords))
        mask_st = mask.reshape(len(space_coords), len(time_coords))


        y_st = y_st.T

        # convert to pandas
        df = pd.DataFrame(y_st)
        df.index = pd.to_datetime(df.index)
        super().__init__(dataframe=df, similarity_score="distance")


# class SoilMoistureSplitter(FixedIndicesSplitter):
#     def __init__(self):
#         # train_idxs = []
#         # valid_idxs = []
#         # for i in range(36):
#         #     for j in range(365):
#         #         train_idxs.append(i*365 + j)
#         #
#         #     start = 10
#         #     end = 50
#         #
#         #     for j in range(start, end):
#         #         valid_idxs.append(i*365 + j)
#         #
#         #
#         # test_idxs = []
#         # for i in range(36):
#         #     for j in range(365):
#         #         test_idxs.append(i*365 + 365 + j)
#
#         train_idxs = [0, 1, 2]
#         valid_idxs = [3, 4, 5]
#         test_idxs = [6, 7, 8]
#
#         super().__init__(train_idxs, valid_idxs, test_idxs)


class SoilMoistureSplitter(Splitter):

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


        self.set_indices(idx[:val_start - dataset.samples_offset],
                         idx[val_start:test_start - dataset.samples_offset],
                         idx[test_start:])

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--val-len', type=float or int, default=0.2)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser


if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values
    dataset = SoilMoistureSparse()
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
