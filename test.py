from datasets.dataset import MakeValidSet, Kodak24, MakeNCD
from invertible_net import Inveritible_Colorization
from datasets.utls import str2bool
import torch.utils.data as data
from utlz import Quantization

import os
import torch, cv2
import argparse
import numpy as np
from termcolor import colored
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt


def main(args):
    device = args.device

    # ValidSet = MakeValidSet(full=True)
    ValidSet = Kodak24()
    ValidLoader = data.DataLoader(ValidSet, batch_size=8, num_workers=0, shuffle=False, pin_memory=True)

    net = Inveritible_Colorization()

    net.load_state_dict(torch.load("./models/ColorFlow_IDN.pth"))

    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.cuda(device=device)
    quantize = Quantization()

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', str(num))

    net.eval()
    validSamples = 0

    ColorPSNR = 0
    GrayPSNR = 0

    for batch_idx, (tensor_g, tensor_c, h, w) in enumerate(ValidLoader):
        n, _, _, _ = tensor_c.size()

        tensor_g, tensor_c = tensor_g.cuda(device=device), tensor_c.cuda(device=device)

        tensor_g.requires_grad = False
        tensor_c.requires_grad = False

        validSamples += tensor_c.size(0)
        with torch.no_grad():
            tensor_pg = net(x=[tensor_c], rev=False)[0][:, [0], :, :]
            tensor_pg = quantize(tensor_pg)
            z = torch.randn(size=(n, 2, 512, 512)).to(tensor_pg.device) * 1 / 255
            tensor_x = torch.cat((tensor_pg, z), dim=1)
            tensor_pc = net(x=[tensor_x], rev=True)[0]
        tensor_pg = tensor_pg.squeeze().cpu().detach().numpy()
        tensor_c = tensor_c.squeeze().cpu().detach().numpy()
        tensor_pc = tensor_pc.squeeze().cpu().detach().numpy()
        tensor_g = tensor_g.squeeze().cpu().detach().numpy()
        for i in range(n):
            img_colorized = tensor_pc[i].transpose((1, 2, 0)).clip(0, 1)
            img_ground_th = tensor_c[i].transpose((1, 2, 0)).clip(0, 1)
            img_colorized = img_colorized[:h[i], :w[i], :]
            img_ground_th = img_ground_th[:h[i], :w[i], :]
            # img_colorized = cv2.resize(img_colorized, dsize=(h[i], w[i]), interpolation=cv2.IMREAD_ANYDEPTH)
            # img_ground_th = cv2.resize(img_ground_th, dsize=(h[i], w[i]), interpolation=cv2.IMREAD_ANYDEPTH)

            psnr_c = round(compare_psnr(img_colorized, img_ground_th, data_range=1), 4)

            img_grayscale = tensor_pg[i].clip(0, 1)
            img_grayed_th = tensor_g[i].clip(0, 1)
            img_grayscale = img_grayscale[:h[i], :w[i]]
            img_grayed_th = img_grayed_th[:h[i], :w[i]]
            # img_grayscale = cv2.resize(img_grayscale, dsize=(h[i], w[i]), interpolation=cv2.IMREAD_ANYDEPTH)
            # img_grayed_th = cv2.resize(img_grayed_th, dsize=(h[i], w[i]), interpolation=cv2.IMREAD_ANYDEPTH)
            psnr_g = round(compare_psnr(img_grayscale, img_grayed_th, data_range=1), 4)

            print("#sample:" + str(batch_idx) + "_" + str(i) +
                  ' Colorized => PSNR: %.4f' % psnr_c, end=" | ")
            print('Grayscale => PSNR: %.4f' % psnr_g)

            # cv2.imwrite(os.path.join("./outputs/", "kodim" + str(i+1) + "_our_restored.png"),
            #             np.uint8(img_colorized[:, :, ::-1] * 255))
            # cv2.imwrite(os.path.join("./outputs/", "kodim" + str(i+1) + "_our_grayscale.png"),
            #             np.uint8(img_grayscale * 255))

            ColorPSNR += psnr_c
            GrayPSNR += psnr_g

    ColorPSNR = ColorPSNR / validSamples
    GrayPSNR = GrayPSNR / validSamples

    print("number of samples:" + str(validSamples))
    print(colored('Color PSNR = %.4f', 'red') % ColorPSNR)
    print(colored('Gray PSNR = %.4f', 'red') % GrayPSNR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--root", type=str, default="/media/ruizhao/programs/datasets/Colorization/NCDataset/",
                        help="data root")
    # For Training
    parser.add_argument("--load_checkpoint", type=str2bool, default=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    main(args)
