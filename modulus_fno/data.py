import os
import random
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class MinMaxScaler:
    def __init__(self, data, min_=0, max_=1) -> None:
        data = data.astype(np.float64)
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.min_ = min_
        self.max_ = max_

    def transform(self, x):
        d_diff = self.data_max - self.data_min
        mask = d_diff == 0
        d_diff[mask] = 1
        s_diff = self.max_ - self.min_

        res = (x - self.data_min) / d_diff * s_diff + self.min_
        return res.astype(np.float32)

    def inverse_transform(self, x):
        d_diff = self.data_max - self.data_min
        s_diff = self.max_ - self.min_
        return (x - self.min_) / s_diff * d_diff + self.data_min


class ChannelMinMaxScaler(MinMaxScaler):
    def __init__(self, data, axis_apply, min_=0, max_=1) -> None:
        super().__init__(data, min_, max_)
        data = data.astype(np.float64)
        self.data_min = np.nanmin(data, axis=axis_apply, keepdims=True)
        self.data_max = np.nanmax(data, axis=axis_apply, keepdims=True)


class SOMAdata(Dataset):
    def __init__(
        self, path, mode, time_steps_per_forward=30, transform=False, train_noise=False
    ):
        """path: the hd5f file path, can be relative path
        mode: ['trian', 'val', 'test']
        """
        super(SOMAdata, self).__init__()
        self.mode = mode
        self.train_noise = train_noise

        DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(DIR, path)
        self.data = h5py.File(data_path, "r")
        keys = list(self.data.keys())

        random.Random(0).shuffle(keys)
        TRAIN_SIZE = int(0.6 * len(keys))
        TEST_SIZE = int(0.1 * len(keys))

        self.time_steps_per_forward = time_steps_per_forward

        sample_data = self.data["forward_0"][...]
        # print(sample_data.shape)

        self.mask1 = sample_data < -1e16
        self.mask2 = sample_data > 1e16

        sample_data[self.mask1] = np.nan
        sample_data[self.mask2] = np.nan

        self.scaler = ChannelMinMaxScaler(sample_data, (0, 1, 2))
        self.transform = transform

        # create a mask for loss calculation
        self.loss_mask = np.logical_or(self.mask1, self.mask2)[
            0, :, :, 0
        ]  # mask only in x,y plane thus size of [100, 100] this will broadcast in element wise product
        self.loss_mask = np.array(~self.loss_mask, dtype=int)  # True - 0; False - 1
        # self.loss_mask = np.transpose(self.loss_mask, axes=[3, 0, 1, 2])[:-1, ...]
        # self.loss_mask = np.expand_dims(self.loss_mask, axis=0) # expand batch dimension for broadcasting
        self.loss_mask = torch.from_numpy(self.loss_mask).float()

        if mode == "train":
            self.keys = keys[:TRAIN_SIZE]
        elif mode == "val":
            self.keys = keys[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
        elif mode == "test":
            self.keys = keys[-TEST_SIZE:]
            print("Test set keys", self.keys)
        else:
            raise Exception(
                f'Invalid mode: {mode}, please select from "train", "val", and "test".'
            )

    def preprocess(self, x, y):
        """Prepare data as the input-output pair for a single forward run
        x has the shape of (60, 100, 100, 17)
        the goal is to first move the ch axis to the second -> (17, 100, 100)
        then create input output pair where the input shape is (17, 100, 100) and the output shape is (16, 100, 100)
        idx 14 is the varying parameter for the input.

        """
        assert len(x.shape) == 3, "Incorrect data shape!"

        x[
            self.mask1[0]
        ] = 0  # every field has the same mask so use the first one and keep the dimension.
        x[self.mask2[0]] = 0
        y[self.mask1[0]] = 0
        y[self.mask2[0]] = 0

        if self.transform:
            d = np.stack((x, y), axis=0)
            d = self.scaler.transform(d)
            x = d[0]
            y = d[1]
        x_in = np.transpose(x, axes=[2, 0, 1])
        x_out = np.transpose(y, axes=[2, 0, 1])[:-1, ...]

        if self.train_noise and self.mode != "test":
            noise = np.random.normal(
                loc=0.0, scale=3e-4, size=x_in.shape
            )  # http://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf
            x_in = x_in + noise
        return (x_in, x_out)

    def __len__(self):
        return len(self.keys) * (
            self.time_steps_per_forward - 1
        )  # b/c n  time steps can create n-1 input-output pairs

    def __getitem__(self, index):
        # get the key idx
        key_idx = int(index / (self.time_steps_per_forward - 1))
        in_group_idx = index % (self.time_steps_per_forward - 1)
        data_x = self.data[self.keys[key_idx]][in_group_idx]
        data_y = self.data[self.keys[key_idx]][in_group_idx + 1]
        x, y = self.preprocess(data_x, data_y)
        assert not np.any(np.isnan(x)) and not np.any(
            np.isnan(y)
        ), "Data contains NaNs!!!"
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


if __name__ == "__main__":
    data = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "train"
    )
