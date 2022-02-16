# Written by Sadhana Ravikumar
# Script used for running successive over relaxation model


import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class MoveLaplaceToSeg(AbstractTransform):
        '''
        data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
        '''
        def __init__(self,channel_id, key_origin="data", key_target="seg", remove_from_origin = True):
            self.remove_from_origin = remove_from_origin
            self.key_target = key_target
            self.key_origin = key_origin
            self.channel_id = channel_id

        def __call__(self, **data_dict):
            origin = data_dict.get(self.key_origin)
            target = data_dict.get(self.key_target)
            laplace = origin[:,self.channel_id]
            laplace = np.expand_dims(laplace, 1)
            target = np.concatenate((target, laplace), 1)
            data_dict[self.key_target] = target

            if self.remove_from_origin:
                remaining_channels = [i for i in range(origin.shape[1]) if i != self.channel_id]
                origin = origin[:, remaining_channels]
                data_dict[self.key_origin] = origin

            return data_dict

