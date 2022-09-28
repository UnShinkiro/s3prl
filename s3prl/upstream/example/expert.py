from collections import OrderedDict
from typing import Dict, List, Union

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
        ckpt = torch.load(ckpt, map_location="cpu")
        config = ckpt["config"]

        self.preprocessor, feat_dim = create_transform(config["data"]["audio"])
        self.name = "[Example UpstreamExpert]"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )

        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        self.model1 = nn.Linear(1, HIDDEN_DIM)
        self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

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
        print(features.shape())
        print("\n\n\n\n")
        feat_lengths = torch.LongTensor(feat_lengths)

        from julia import Main
        Main.eval('using Pkg; Pkg.activate("NODE-APC")')
        Main.using("Flux")
        Main.using("BSON: @load")
        Main.using("Random")
        Main.eval('@load "NODE-APC/360hModel.bson" trained_model post_net')

        Main.data = features
        Main.eval('data = Float32.(data)')
        Main.eval('Flux.reset!(trained_model)')
        feature = Main.eval('feature = trained_model(data)')
        hidden = Main.eval('hidden = post_net(feature)')
        # hidden: (batch_size, max_len, hidden_dim)
        # wavs: (batch_size, max_len, 1)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [hidden, feature],
            "PR": [hidden, feature],
            "ASR": [hidden, feature],
            "QbE": [hidden, feature],
            "SID": [hidden, feature],
            "ASV": [hidden, feature],
            "SD": [hidden, feature],
            "ER": [hidden, feature],
            "SF": [hidden, feature],
            "SE": [hidden, feature],
            "SS": [hidden, feature],
            "secret": [hidden, feature],
        }
