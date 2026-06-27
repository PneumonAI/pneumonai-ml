import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch import nn


def generate_heatmap(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Return a Grad-CAM heatmap explaining the pneumonia class prediction.

    Always explains class 1 (pneumonia) regardless of the predicted label,
    so a 'normal' result still shows where the model looked for opacity.

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
