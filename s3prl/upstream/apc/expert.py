from julia import Main
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence

from ..interfaces import UpstreamBase
from .audio import create_transform


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        ckpt = torch.load(ckpt, map_location="cpu")
        config = ckpt["config"]

        self.preprocessor, feat_dim = create_transform(config["data"]["audio"])
       	"""
	self.model = APC(feat_dim, **config["model"]["paras"])
        self.model.load_state_dict(ckpt["model"])

        if len(self.hooks) == 0:
            self.add_hook(
                "self.model.rnn_layers[1]",
                lambda input, output: pad_packed_sequence(input[0], batch_first=True)[
                    0
                ],
            )
            self.add_hook(
                "self.model.rnn_layers[2]",
                lambda input, output: pad_packed_sequence(input[0], batch_first=True)[
                    0
                ],
            )
            self.add_hook("self.model", lambda input, output: output[1])
	"""

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]

        features = pad_sequence(features, batch_first=True)
        print(features.cpu().numpy())
        print("\n\n\n\n")
        feat_lengths = torch.LongTensor(feat_lengths)

        Main.eval('using Pkg; Pkg.activate("/home/z5195063/master/NODE-APC")')
        Main.using("Flux")
        Main.using("BSON: @load")
        Main.using("Random")
        Main.eval('@load "/home/z5195063/master/NODE-APC/360hModel.bson" trained_model post_net')

        print(Main.eval('trained_model'))
        Main.data = features.cpu().numpy()
        Main.eval('data = Float32.(data)')
        Main.eval('data = reshape(data, (80,:))')
        Main.eval('print(size(data))')
        Main.eval('Flux.reset!(trained_model)')
        feature = Main.eval('feature = trained_model(data)')
        hidden = Main.eval('hidden = post_net(feature)')
        # hidden: (batch_size, max_len, hidden_dim)
        # feature: (batch_size, max_len, hidden_dim)
        feature = feature.reshape(1,-1,512)
        hidden = hidden.reshape(1,-1,80)

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
