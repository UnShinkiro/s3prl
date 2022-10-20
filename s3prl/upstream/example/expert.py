from julia import Main
from collections import OrderedDict
from typing import Dict, List, Union

import torch
import pickle
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .audio import create_transform

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        file = open('/home/z5195063/master/config.pkl', 'rb')
        config = pickle.load(file)
        file.close()

        self.preprocessor, feat_dim = create_transform(config["data"]["audio"])
        self.name = "[Example UpstreamExpert]"

        Main.eval('using Pkg; Pkg.activate("/home/z5195063/master/NODE-APC")')
        Main.using("Flux")
        Main.using("BSON: @load")
        Main.using("CUDA")
        Main.using("Random")
        Main.eval('@load "/srv/scratch/z5195063/360hModel_v3.bson" trained_model post_net')
        Main.eval('trained_model |> gpu')

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )

        # The model needs to be a nn.Module for finetuning, not required for representation extraction

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 160

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]

        features = pad_sequence(features, batch_first=True)
        feat_lengths = torch.LongTensor(feat_lengths)
        batch_size = (features.size()[0])
        features = features.cpu().numpy()
        
        ret_feature = []
        for file_idx in range(np.shape(features)[0]):
            Main.eval('Flux.reset!(trained_model)')
            Main.data = features[file_idx,:,:]
            Main.eval('data = Float32.(data)')
            Main.eval('print(size(data))')
            Main.eval('data = [data[frame_idx,:] for frame_idx=1:size(data)[1]]')
            Main.eval('print(size(data))')
            feature = Main.eval('feature = trained_model.(data)')
            ret_feature.append(feature)
        
        ret_feature = np.asarray(ret_feature)
        ret_feature = torch.from_numpy(ret_feature).cuda()

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [ret_feature, ret_feature],
            "PR": [ret_feature, ret_feature],
            "ASR": [ret_feature, ret_feature],
            "QbE": [ret_feature, ret_feature],
            "SID": [ret_feature, ret_feature],
            "ASV": [ret_feature, ret_feature],
            "SD": [ret_feature, ret_feature],
            "ER": [ret_feature, ret_feature],
            "SF": [ret_feature, ret_feature],
            "SE": [ret_feature, ret_feature],
            "SS": [ret_feature, ret_feature],
            "secret": [ret_feature, ret_feature],
        }
