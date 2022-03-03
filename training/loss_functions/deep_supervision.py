#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn
import numpy as np
import nibabel as nib 
#from nnunet.utilities.one_hot_encoding import to_one_hot
import torch.nn.functional as F


def doublesigmoid_threshold(data, lower_lim, upper_lim):

    steepness = 10

    lower_thresh = 1/(1 + torch.exp(-steepness*(data - lower_lim)))
    upper_thresh = 1/(1 + torch.exp(steepness*(data - upper_lim)))

    output = torch.mul(lower_thresh,upper_thresh)
    output = output.squeeze()

    return output

def convert_laplacian_toseg(data):

    thresholds = [-0.1,0.1, 0.3, 0.5, 0.7, 0.9]
    result = torch.zeros((data.shape[0],len(thresholds), *data.shape[2:]), dtype=data.dtype)
    for i, l in enumerate(thresholds):
        output = doublesigmoid_threshold(data, l,l+0.3)
        result[:,i] = output

    return result

def to_one_hot(seg, all_seg_labels=None):

    if all_seg_labels is None:
        all_seg_labels = torch.unique(seg) #np.unique
    result = torch.zeros((seg.shape[0],len(all_seg_labels), *seg.shape[2:]), requires_grad = True, dtype=seg.dtype).cuda() # np.zeros
    for i, l in enumerate(all_seg_labels):
        result[:,i][seg[:,0] == l] = 1

    return result

def convert_laplacian_toseg(data, thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]):

    labels = torch.arange(1,6).int()
    for i in range(len(thresholds)-1):
    
        lower_thresh = thresholds[i]
        upper_thresh = thresholds[i+1]
        if upper_thresh != 1:
            data[(data > lower_thresh) & (data <= upper_thresh)] = labels[i]
        else:
            data[(data > lower_thresh) & (data < upper_thresh)] = labels[i]

    #Remove noisy values
    data[data < 0] = 0

    #Make sure all values are whole numbers
    data = torch.round(data)

    return data


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

#Added by SR for computing MSE loss with SOR module
class MultipleOutputLoss2_SOR(nn.Module):
    def __init__(self, loss, weight_factors=None, sor_start_epoch = 10, lambda_weight = 0.5):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        :param lambda_weight: weighting between segmentation loss and thickness loss
        """
        super(MultipleOutputLoss2_SOR, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.lambda_weight = lambda_weight

        #MSE loss - first try without averaging over number of gm pixels
        self.mse_loss = nn.MSELoss()
        self.sor_start_epoch = sor_start_epoch

        self.bnd = 15

    def update_lambda(self, lambda_weight):
        self.lambda_weight = lambda_weight

    def forward(self, x, y, epoch):

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"


        #Extract thickness prediction from x and compute mse loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        if epoch > self.sor_start_epoch:
            gm_mask = y[1]
            
            #Mask the target incase of interpolation errors
            r = np.random.randint(10)
            #output_file = '/data/sadhanar/pred_laplace' + str(r) + '.nii.gz'
            #laplace = torch.swapaxes(x[0].squeeze(),0,3)
            #nib.save(nib.Nifti1Image(laplace.detach().cpu().numpy(), np.eye(4)),output_file)

            #Only include elements in the ground truth gm
            t_pred = x[0]
            t_pred[gm_mask != 1] = 0
            #t_pred = convert_laplacian_toseg(t_pred)

            t_gt = y[0]
            t_gt[gm_mask != 1] = 0
            #t_gt = convert_laplacian_toseg(t_gt)
 
            #Crop the boundaries of laplacian maps due to boundary errors
            t_gt = t_gt[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]
            t_pred = t_pred[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]
            gm_mask = gm_mask[:,:,self.bnd:-self.bnd, self.bnd:-self.bnd, self.bnd:-self.bnd]

            output_file = '/data/sadhanar/groundtruth' + str(r) + '.nii.gz'
            laplace = torch.swapaxes(t_gt.squeeze(),0,3)
            nib.save(nib.Nifti1Image(laplace.cpu().numpy(), np.eye(4)),output_file)

            output_file = '/data/sadhanar/pred' + str(r) + '.nii.gz'
            laplace = torch.swapaxes(t_pred.squeeze(),0,3)
            nib.save(nib.Nifti1Image(laplace.detach().cpu().numpy(), np.eye(4)),output_file)
       
            #t_pred = to_one_hot(t_pred, all_seg_labels = [0,1,2,3,4,5])

            #Mask by ground truth segmentation
            t_pred = t_pred[gm_mask == 1]
            t_gt = t_gt[gm_mask==1]

            # Compute average MSE loss over pixels included in the gm.
            t_loss = self.lambda_weight*self.mse_loss(t_pred,t_gt)
            
            #Use MSE loss. Need to find a differentiable way to convert laplacian map to seg
            #Instead of MSE loss, convert laplacian maps to segmentations and then use segmentation loss
            #self.t_loss = self.loss(t_pred,t_gt)
            x = x[1:]

        # First element is the ground truth laplacian map 
        y = y[1:]

        if self.weight_factors is None:
            weights = [1] * (len(x) - 1) # since first element in x is the thickness map
        else:
            weights = self.weight_factors

        # Convert the hippocampus label in the ground truth to gray matter for dice loss
        gt = y[0]
        gt[gt == 5] = 1

        l = weights[0] * self.loss(x[0], gt)
        
        for i in range(1, len(x)):
            # Idexing of weights and y is off by one because I already removed the laplacian map
            if weights[i] != 0:
                gt = y[i]
                gt[gt == 5] = 1
                l += weights[i] * self.loss(x[i], gt)

        #Extract thickness prediction from x and compute mse loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        if epoch > self.sor_start_epoch:
            loss_array = torch.tensor([l, t_loss])
            l = l + t_loss
        else:
            loss_array = torch.tensor([l, 0])

        return l, loss_array
