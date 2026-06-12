import torch


class TanhScaler:
    def __init__(self, const=5000):
        self.constant = const

    def scale(self, y_train):
        return torch.tanh(y_train/self.constant)

    def reverse(self, scaled_tensor):
        return self.constant * torch.atanh(scaled_tensor)


class AsinhScaler:
    def __init__(self, const=7000):
        self.constant = const

    def scale(self, y_train):
        return torch.asinh(y_train/self.constant)

    def reverse(self, scaled_tensor):
        return self.constant * torch.sinh(scaled_tensor)


class DistribScaler:
    def __init__(self, mu, sigma):
        self.mean = mu
        self.std = sigma

        if self.std == 0:
            self.std = torch.tensor(1.0)

    def scale(self, y_train):
        return (y_train - self.mean)/self.std

    def reverse(self, scaled_tensor):
        return scaled_tensor*self.std + self.mean
