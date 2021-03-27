import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, Nclasses):
        super(DiceLoss, self).__init__()
        self.Nclasses = Nclasses

    def _one_hot_encoder(self, input_tensor):
        tensorList = []
        for i in range(self.Nclasses):
            temp_prob = input_tensor == i
            tensorList.append(temp_prob.unsqueeze(1))
        outputTensor = torch.cat(tensorList, dim=1)
        return ( outputTensor.float() )

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum, z_sum = torch.sum(target * target), torch.sum(score * score)
        diceLoss = 1- (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return ( diceLoss )

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.Nclasses
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.Nclasses):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return ( loss / self.Nclasses )


def calculate_metric_percase(pred, gt):
	prediction = pred
	groundTruth = gt
    prediction[prediction > 0] = 1
    groundTruth[groundTruth > 0] = 1
    if prediction.sum() > 0 and groundTruth.sum()>0:
        dice = metric.binary.dc(prediction, groundTruth)
        hd95 = metric.binary.hd95(prediction, groundTruth)
        return ( dice, hd95 )
    elif prediction.sum() > 0 and groundTruth.sum()==0:
        return ( 1, 0 )
    else:
        return ( 0, 0 )
