"""Transform .dcm files into readable tensor format for ReSNet """
from torch import Tensor, from_numpy
from torch.nn.functional import interpolate 
import numpy as np
import pydicom
from pathlib import Path
from .specification import PreprocessingSpec

def preprocess(path : Path, spec : PreprocessingSpec) -> Tensor:
    dicom = pydicom.dcmread(path)
    array = dicom.pixel_array
    photometric_interpretation  = dicom.PhotometricInterpretation 
    if photometric_interpretation == "MONOCHROME1" and spec.invert_monochrome1:
        array = np.iinfo(array.dtype).max - array
    array = array.astype(np.float32)
    if array.max() == array.min():
        array = np.zeros_like(array)
        
    else:
        array = (array - array.min()) / (array.max() - array.min())

    tensor = from_numpy(array)[None, None, :, :]
    tensor = interpolate(tensor, size=(spec.height, spec.width), mode="bilinear", align_corners=False)
    tensor = tensor.squeeze()
    tensor = tensor.unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1)
    tensor_mean = Tensor(spec.mean).view(3, 1, 1)
    tensor_std = Tensor(spec.std).view(3, 1, 1)
    tensor = (tensor - tensor_mean) / tensor_std 
    return tensor