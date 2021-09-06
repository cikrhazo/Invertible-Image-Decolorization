from datasets.dataset import MakeTrainSet, Kodak24
from utlz import Quantization
from invertible_net import Inveritible_Decolorization
from datasets.utls import str2bool
import torch.utils.data as data
from criteria import ConsistencyLoss
from torch.optim import Adam
import os, sys, time, math
import torch, cv2
import logging
import argparse
import numpy as np
import torch.nn as nn
from termcolor import colored
from mmcv.utils import get_logger
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(args):
    beginner = args.beginner
    stride = args.stride
    device = args.device
    batch_size = args.batch_size
    epoch = args.Epoch
    lr = args.lr
    weight_decay = args.weight_decay
    model_path = "./models/"

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(model_path, f'{timestamp}.log')
    logger = get_logger(name='IDN', log_file=log_file, log_level=logging.INFO)
    logger.info(f"batch size {batch_size}")

    TrainSet = MakeTrainSet()
    TrainLoader = data.DataLoader(TrainSet, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    ValidSet = Kodak24()
    ValidLoader = data.DataLoader(ValidSet, batch_size=8, num_workers=8, shuffle=False, pin_memory=True)

    logger.info("# Training Samples: " + str(TrainSet.__len__()) + "; Valid Samples: " + str(ValidSet.__len__()))

    net = Inveritible_Decolorization()
    optimizer = Adam(net.parameters(), weight_decay=weight_decay, betas=(0.5, 0.999), lr=lr)

    if args.load_checkpoint:
        logger.info("loading checkpoint...")
        net.load_state_dict(torch.load("./models/****.pth"))

    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.cuda(device=device)
    quantize = Quantization()
    loss_cons = ConsistencyLoss(device="cuda:0", img_shape=(batch_size, 1, 128, 128), c_weight=args.c_weight)
    loss_dist = nn.MSELoss(reduction="sum")

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', str(num))

    BestPSNR = 0.
    
    # use warmup to stablize the training, or it may not converage

    for i in range(beginner, epoch):
        loss_record = 0
        trainSamples = 0

        if i != beginner:
            TrainSet = MakeTrainSet()
            TrainLoader = data.DataLoader(TrainSet, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)

        for batch_idx, (tensor_g, tensor_c) in enumerate(TrainLoader):
            n, _, h, w = tensor_c.size()

            tensor_c = tensor_c.cuda(device=device)
            tensor_g = tensor_g.cuda(device=device)

            tensor_g.requires_grad = False
            tensor_c.requires_grad = False

            trainSamples += n

            net.train(True)

            optimizer.zero_grad()

            tensor_x = net(x=[tensor_c], rev=False)[0]
            tensor_prg = tensor_x[:, [0], :, :]
            tensor_z = tensor_x[:, 1:, :, :]

            tensor_z_ = torch.randn_like(tensor_z).to(tensor_z.device)
            tensor_g_ = quantize(tensor_prg)
            tensor_x_ = torch.cat((tensor_g_, tensor_z_), dim=1)
            tensor_y = net(x=[tensor_x_], rev=True)[0]

            loss_invt = loss_cons(gray_img=tensor_prg, ref_img=tensor_g, original_img=tensor_c,
                                  restored_img=tensor_y, loss_stage=1, s_wegiht=args.s_weight) / n
            loss_self = (tensor_z**2).sum() / (128**2 * 2 * n)
            loss_gray = loss_dist(tensor_prg, tensor_g) / n
            loss = loss_invt + args.r_weight * loss_self / 2 + args.g_weight * loss_gray

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            loss_record += loss.detach().cpu() * n
            if batch_idx % 10 == 9 or batch_idx == 0:
                print('#batch: %4d; learning rate: %.4e; loss_invt: %3.4f; loss_self: %3.4f; loss_gray: %3.4f.'
                      % (batch_idx + 1, optimizer.param_groups[0]['lr'],
                         float(loss_invt), float(loss_self), float(loss_gray)))
        avg_loss = loss_record / trainSamples
        logger.info('Train: Epoch = %3d | Ave Loss = %.4f | Train Samples = %3d.' % (i + 1, avg_loss, trainSamples))

        if i % stride == (stride - 1):
            torch.save(net.module.state_dict(), os.path.join(model_path, 'net_epoch_' + str(i + 1).zfill(3) + '.pth'))

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
            for ii in range(n):
                img_colorized = tensor_pc[ii].transpose((1, 2, 0)).clip(0, 1)[:h[ii], :w[ii], :]
                img_ground_th = tensor_c[ii].transpose((1, 2, 0)).clip(0, 1)[:h[ii], :w[ii], :]
                psnr_c = round(compare_psnr(img_ground_th, img_colorized, data_range=1), 4)

                img_grayscale = tensor_pg[ii].clip(0, 1)[:h[ii], :w[ii]]
                img_grayed_th = tensor_g[ii].clip(0, 1)[:h[ii], :w[ii]]
                psnr_g = round(compare_psnr(img_grayed_th, img_grayscale, data_range=1), 4)

                print("#sample:" + str(batch_idx) + "_" + str(ii) +
                      ' Colorized => PSNR: %.4f' % (psnr_c), end=" | ")
                print('Grayscale => PSNR: %.4f' % (psnr_g))

                ColorPSNR += psnr_c
                GrayPSNR += psnr_g

        ColorPSNR = ColorPSNR / validSamples
        GrayPSNR = GrayPSNR / validSamples

        ValidPSNR = (ColorPSNR + GrayPSNR) / 2
        if BestPSNR < ValidPSNR:
            BestPSNR = ValidPSNR
            torch.save(net.module.state_dict(), os.path.join(model_path, 'ColorFlow.pth'))
        logger.info(
            'Valid Samples = %3d | ' % validSamples
            + colored('Color PSNR = %.4f', 'red') % ColorPSNR + " | "
            + colored('Best PSNR = %.4f', 'green') % BestPSNR + " | "
            + 'Gray PSNR = %.4f' % GrayPSNR
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--stride", type=int, default=10, help='the stride for saving models')
    parser.add_argument("--paired", type=str2bool, default=True, help="paired or unpaired")
    parser.add_argument("--root", type=str, default="F:/datasets/Colorization/VCIP2020_Colorization_Challenge/",
                        help="data root")
    # For Training
    parser.add_argument("--load_checkpoint", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--beginner", type=int, default=0)
    parser.add_argument('--Epoch', type=int, default=1)
    parser.add_argument("--c_weight", type=float)
    parser.add_argument("--s_weight", type=float)
    parser.add_argument("--r_weight", type=float)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    main(args)
