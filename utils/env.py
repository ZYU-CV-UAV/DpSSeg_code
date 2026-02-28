import torch
import platform


def collect_env_info():

    info = {}

    info["OS"] = platform.platform()
    info["Python"] = platform.python_version()
    info["CUDA available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["CUDA version"] = torch.version.cuda
        info["cuDNN version"] = torch.backends.cudnn.version()

    info["PyTorch"] = torch.__version__

    return info