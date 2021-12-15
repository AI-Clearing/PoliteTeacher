from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch import nn

class RCNN_CPS(nn.Module):
    """CPS network wrapper"""
    def __init__(self, branch1, branch2):
        super(RCNN_CPS, self).__init__()
        self.branches = nn.ModuleList([branch1, branch2])

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branches[0](data)
            return pred1
        return self.branches[step-1](data)