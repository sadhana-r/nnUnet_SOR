# Written by Sadhana Ravikumar
# Script used for running successive over relaxation model


import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np

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

class MoveLaplaceToSeg(AbstractTransform):
        '''
        data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
        '''
        def __init__(self,channel_id, key_origin="data", key_target="seg", remove_from_origin = True, laplace_seg = False):
            self.remove_from_origin = remove_from_origin
            self.key_target = key_target
            self.key_origin = key_origin
            self.channel_id = channel_id

        def __call__(self, **data_dict):
            origin = data_dict.get(self.key_origin)
            target = data_dict.get(self.key_target)
            laplace = origin[:,self.channel_id]
            laplace = np.expand_dims(laplace, 1)

            if laplace_seg:
                #Convert to seg one hot
                laplace_onehot = convert_laplacian_toseg(laplace)
                laplace_multilabel = laplace_onehot.argmax(1)
                target = np.concatenate((target, laplace_multilabel), 1)
            
            else:
                target = np.concatenate((target, laplace), 1)
            
            data_dict[self.key_target] = target

            if self.remove_from_origin:
                remaining_channels = [i for i in range(origin.shape[1]) if i != self.channel_id]
                origin = origin[:, remaining_channels]
                data_dict[self.key_origin] = origin

            return data_dict

