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
        Main.eval('print("Benchmarking with NODE-APC")')
        Main.eval('print')
        Main.using("Flux")
        Main.using("BSON: @load")
        Main.using("CUDA")
        Main.using("Random")
        Main.using("DiffEqFlux")
        Main.using("DifferentialEquations")
        Main.eval('@load "/srv/scratch/z5195063/devNODEModel.bson" prenet trained_model post_net')
        Main.eval('lspan = (0.0f0,1.0f0)')
        Main.eval('node = NeuralODE(trained_model,lspan,Tsit5(),save_start=false,saveat=1,reltol=1e-7,abstol=1e-9)')
        Main.eval('node_apc = Chain(prenet, node)')
        #Main.eval('trained_model = trained_model |> gpu')

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
        features = features.cpu().numpy()
        
        print('Before: ', np.shape(features))
        input_dim = np.shape(features)[-1]
        length = np.shape(features)[-2]
        features = np.reshape(features, (-1,length,input_dim))
        print('After: ', np.shape(features))
        ret_feature = []
        for count, file in enumerate(features):
            Main.eval('Flux.reset!(trained_model)')
            Main.data = file
            Main.eval('data = Float32.(data)')
            Main.eval(f'data = [data[frame_idx,:] for frame_idx=1:{feat_lengths[count].item()}]')
            #Main.eval('print(typeof(data))')
            #Main.eval('data1 = cu.(data)')
            #Main.eval('print(typeof(data1))')
            #Main.eval('CUDA.allowscalar(true)')
            #Main.eval('data = data |> gpu')
            feature = Main.eval(f'feature = [idx <= size(data)[1] ? node_apc((data[idx])) : zeros(Float32,512) for idx=1:{length}]')
            feature = np.reshape(feature, (length,512))
            print(np.shape(feature))
            ret_feature.append(feature)
        
        ret_feature = np.asarray(ret_feature)
        print(np.shape(ret_feature))
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
