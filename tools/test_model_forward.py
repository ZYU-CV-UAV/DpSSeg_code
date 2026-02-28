import torch
from models.dpseg import DpSSeg


def main():

    model = DpSSeg(num_classes=2).cuda()
    model.eval()

    x = torch.randn(1, 3, 256, 256).cuda()

    with torch.no_grad():
        main_out, aux_outs = model(x)

    print("Main output shape:", main_out.shape)

    for i, aux in enumerate(aux_outs):
        print(f"Aux {i} shape:", aux.shape)


if __name__ == "__main__":
    main()