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
    def __init__(self, loss, weight_factors=None, sor_start_epoch = 10, lambda_weight = 0.2):
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
        self.mse_loss = nn.MSELoss()
        self.sor_start_epoch = sor_start_epoch

    def update_lambda(self, lambda_weight):
        self.lambda_weight = lambda_weight

    def forward(self, x, y, epoch):

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"

        #Extract thickness prediction from x and compute mse loss with ground truth laplacian
        # Need to mask the hippocampus region out of the predicted laplacian map
        if epoch > self.sor_start_epoch:
            mask = y[1]

            #Mask the target incase of interpolation errors
            gm_mask = mask
            gm_mask[gm_mask != 1] = 0 
            t_pred = x[0] * gm_mask
            t_gt = y[0]*gm_mask

            r = np.random.randint(200)
            output_file = '/data/sadhanar/groundtruth' + str(r) + '.nii.gz'
            laplace = torch.swapaxes(t_gt.squeeze(),0,3)
            nib.save(nib.Nifti1Image(laplace.cpu().numpy(), np.eye(4)),output_file)

            output_file = '/data/sadhanar/pred' + str(r) + '.nii.gz'
            laplace = torch.swapaxes(t_pred.squeeze(),0,3)
            nib.save(nib.Nifti1Image(laplace.cpu().numpy(), np.eye(4)),output_file)

            t_loss = self.mse_loss(t_pred,y[0]*gm_mask)
            x = x[1:]

        # First element is the ground truth laplacian map 
        y = y[1:]

        if self.weight_factors is None:
            weights = [1] * (len(x) - 1) # since first element in x is the thickness map
        else:
            weights = self.weight_factors

        # Convert the hippocampus label to gray matter for dice loss
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
            print("loss function: dice: ", l, " laplace ", t_loss)
            l = (1 - self.lambda_weight)*l + self.lambda_weight*t_loss

        return l
