from collections import OrderedDict
import torch
from torch import nn
from .ssl.util_modules import SplitBatchNorm1d, SplitBatchNorm2d


class WrapperClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_class: int, fix_encoder, num_layer: int = 1, hidden_dim: int = None):
        super(WrapperClassifier, self).__init__()
        self.encoder = encoder
        self.__replace_split_bn_to_bn(self.encoder)
        for p in self.encoder.parameters():
            p.requires_grad = not fix_encoder
            # nn.init.normal_(p, 0.0, 1.0)  # initial random weight
        input_dim = encoder.output_dim

        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        projector_list = []
        for n in range(1, num_layer + 1):
            if n == num_layer:  # last layer
                projector_list.append(
                    (f"fc{n}", nn.Linear(hidden_dim, num_class, bias=True))
                )
            else:
                projector_list.extend([
                    (f"fc{n}", nn.Linear(input_dim, hidden_dim, bias=False)),
                    (f"bn{n}", nn.BatchNorm1d(hidden_dim, hidden_dim)),
                    (f"act{n}", nn.ReLU(inplace=True))
                ])
                input_dim = hidden_dim
        self.classifier = nn.Sequential(OrderedDict(projector_list))

    def forward(self, x):
        return self.classifier(self.encoder(x))

    @staticmethod
    def __replace_split_bn_to_bn(modules: torch.nn.Module):
        for module in modules.modules():
            if isinstance(module, (SplitBatchNorm1d, SplitBatchNorm2d)):
                module.num_splits = 1  # num_splits=1 is normal batch norm
