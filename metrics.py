import numpy as np
import torch
import torch.nn as nn


def r2(true, pred):
    if np.isnan(true).any():
        true = true.reshape(-1,)
        pred = pred.reshape(-1,)
        mask = np.isnan(true)
        true = true[~mask]
        pred = pred[~mask]
    ss_res = np.sum(np.power(true - pred, 2), axis=(0, 2, 3))
    ss_tot = np.sum(np.power(true - np.mean(true, axis=(0, 2, 3), keepdims=True), 2), axis=(0, 2, 3))
    return 1 - ss_res / ss_tot


def anomalyCorrelationCoef(true, pred):
    if np.isnan(true).any():
        true = true.reshape(-1,)
        pred = pred.reshape(-1,)
        mask = np.isnan(true)
        true = true[~mask]
        pred = pred[~mask]

    trueMean = np.mean(true, axis=(0, 2, 3), keepdims=True) 
    trueAnomaly =  true - trueMean
    predAnomaly = pred - trueMean

    cov = np.mean(predAnomaly * trueAnomaly, axis=(0, 2, 3))
    std = np.sqrt(np.mean(predAnomaly ** 2, axis=(0, 2, 3)) * np.mean(trueAnomaly ** 2, axis=(0, 2, 3)))
    return cov / std


class ACCLoss(nn.Module):
    def __init__(self):
        super(ACCLoss, self).__init__()
    def forward(self, true, pred):
        '''
        true and pred have shape (B, 5, 100, 100)
        we should calculate the channel wise scores
        '''
        TrueMean = torch.mean(true, dim=(0, 2, 3), keepdims=True)
        TrueAnomaly = true - TrueMean
        PredAnomaly = pred - TrueMean

        cov = torch.mean(PredAnomaly * TrueAnomaly, dim=(0, 2, 3))
        std = torch.sqrt(torch.mean(PredAnomaly ** 2, dim=(0, 2, 3)) * torch.mean(TrueAnomaly ** 2, dim=(0, 2, 3)))
        return -torch.mean(cov / std) 

class MSE_ACCLoss(nn.Module):
    def __init__(self, alpha):
        super(MSE_ACCLoss, self).__init__()
        self.alpha = alpha
    def forward(self, true, pred):
        '''
        true and pred have shape (B, 5, 100, 100)
        we should calculate the channel wise scores
        '''
        TrueMean = torch.mean(true, dim=(0, 2, 3), keepdims=True)
        TrueAnomaly = true - TrueMean
        PredAnomaly = pred - TrueMean

        cov = torch.mean(PredAnomaly * TrueAnomaly, dim=(0, 2, 3))
        std = torch.sqrt(torch.mean(PredAnomaly ** 2, dim=(0, 2, 3)) * torch.mean(TrueAnomaly ** 2, dim=(0, 2, 3)))

        acc_term = -torch.mean(cov / std)
        mse_term = torch.mean((true - pred) ** 2)

        return self.alpha * mse_term  + (1 - self.alpha) * acc_term
