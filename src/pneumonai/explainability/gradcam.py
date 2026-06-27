import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch import nn


class GradCAMResNet(nn.Module):
    """ResNet wrapper that returns (logits, feature_maps) for C++ Grad-CAM.

    Exposes layer4 activations as a second output so that C++ libtorch can
    compute d(score)/d(feats) via torch::autograd::grad without Python hooks.
    The graph is kept alive — do NOT wrap the forward call in torch.no_grad().
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feats = self.layer4(x)                                    # [1, 512, 7, 7]
        logits = self.fc(torch.flatten(self.avgpool(feats), 1))   # [1, 1]
        return logits, feats


def generate_heatmap(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Return a Grad-CAM heatmap explaining the pneumonia class prediction.

    Python oracle — used for parity validation against C++ Grad-CAM output.
    Always explains class 1 (pneumonia) regardless of the predicted label.

    Args:
        model:  ResNet model in eval mode, single-output binary classifier.
        tensor: Preprocessed input tensor, shape [1, 3, H, W], float32.

    Returns:
        Heatmap as a float32 numpy array, shape [H, W], values in [0.0, 1.0].
    """
    target_layers = [model.layer4[-1]]
    targets = [BinaryClassifierOutputTarget(1)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        heatmap = cam(input_tensor=tensor, targets=targets)

    return heatmap[0].astype(np.float32)
