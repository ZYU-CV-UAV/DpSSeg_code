import torch
import time
from models.dpseg import DpSSeg


def benchmark(model, input_size=(1,3,256,256), runs=200):

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    dummy = torch.randn(input_size).to(device)

    # warmup
    for _ in range(20):
        _ = model(dummy)

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(runs):
        _ = model(dummy)

    torch.cuda.synchronize()

    end = time.time()

    total_time = end - start
    fps = runs / total_time

    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    model = DpSSeg(num_classes=2)
    benchmark(model)