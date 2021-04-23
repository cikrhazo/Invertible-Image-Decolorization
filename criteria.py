import torch.distributions as D
from torch.autograd import Variable
import math

import torch
import torch.nn as nn
from torchvision.models import vgg19


class ConsistencyLoss(nn.Module):
    def __init__(self, device, img_shape, threshold=70 / 127, vgg_layer_idx=21, c_weight=1e-7):
        """
        if vgg_layer_idx=21, it forwards to conv_layer(conv+relu)4_1
        """
        super().__init__()

        self.i_loss = InvertibilityLoss()
        self.g_loss = GrayscaleConformityLoss(device, img_shape, threshold, vgg_layer_idx, c_weight)
        self.q_loss = QuantizationLoss()

    def forward(self, gray_img, ref_img, original_img, restored_img, loss_stage, s_weight):
        i_loss = self.i_loss(original_img, restored_img)

        if loss_stage == 1:
            g_loss = self.g_loss(gray_img, ref_img, original_img, ls_weight=s_weight)
            total_loss = 3 * i_loss + g_loss # 3 channels
        elif loss_stage == 2:
            g_loss = self.g_loss(gray_img, original_img, ls_weight=s_weight)
            q_loss = self.q_loss(gray_img)
            total_loss = i_loss + g_loss + (10 * q_loss)
        else:
            total_loss = 0

        return total_loss


class InvertibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, original_img, restored_img):
        return self.loss(original_img, restored_img)


class GrayscaleConformityLoss(nn.Module):
    def __init__(self, device, img_shape, threshold, vgg_layer_idx, c_weight):
        super().__init__()

        self.threshold = threshold
        self.vgg = nn.DataParallel(vgg19(pretrained=True).features[:vgg_layer_idx]).to(device)
#         self.dis = nn.MSELoss(reduction="sum")
        self.dis = nn.L1Loss(reduction="sum")

        self.c_weight = c_weight
        self.zeros = torch.zeros(img_shape).to(device)

    def lightness(self, gray_img, original_luminance):
        # print(original_luminance.shape, gray_img.shape, self.zeros.shape)
        # loss = torch.mean(torch.max(torch.abs(original_luminance.repeat(1, 3, 1, 1) - gray_img) - self.threshold, self.zeros))
        loss = torch.sum(torch.max(torch.abs(original_luminance - gray_img) - self.threshold, self.zeros))
        return loss

    def contrast(self, gray_img, original_img):
        def _rescale(img):
            img = (img + 1) / 2 * 255
            img[:, 0, :, :] = img[:, 0, :, :] - 123.68      # subtract vgg mean following the implementation by authors(meaning?)
            img[:, 1, :, :] = img[:, 1, :, :] - 116.779
            img[:, 2, :, :] = img[:, 2, :, :] - 103.939
            return img
        # vgg_g = self.vgg(_rescale(gray_img))
        vgg_g = self.vgg(_rescale(gray_img.repeat(1, 3, 1, 1)))
        vgg_o = self.vgg(_rescale(original_img))
        return self.dis(vgg_g, vgg_o)

    def local_structure(self, gray_img, original_luminance):
        def _tv(img):
            h_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
            w_diff = img[:, :, :, 1:] - img[:, :, :, :-1]

            sum_axis = (1, 2, 3)
            var = torch.abs(h_diff).sum(dim=sum_axis) + torch.abs(w_diff).sum(dim=sum_axis)

            return var
        gray_tv = torch.sum(_tv(gray_img) / (128 ** 2))
        original_tv = torch.sum(_tv(original_luminance) / (128 ** 2))
        loss = abs(gray_tv - original_tv)
        return loss

    def forward(self, gray_img, ref_img, original_img, ls_weight):
        # r, g, b = [ch.unsqueeze(1) for ch in torch.unbind(original_img, dim=1)]
        # original_luminance = (.299 * r) + (.587 * g) + (.114 * b)

        original_luminance = ref_img

        l_loss = self.lightness(gray_img, original_luminance)
        c_loss = self.contrast(gray_img, original_img)
        ls_loss = self.local_structure(gray_img, original_luminance)

        return l_loss + (self.c_weight * c_loss) + (ls_weight * ls_loss)


class QuantizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gray_img):
        min_tensor = torch.zeros_like(gray_img).fill_(1)   # fill with maximum value (larger than 255)

        for i in range(0, 256):
            min_tensor = torch.min(min_tensor, torch.abs(gray_img - (i / 127.5 - 1)))

        loss = torch.sum(min_tensor)
        return loss


class FlowLoss(nn.Module):
    def __init__(self):
        super(FlowLoss, self).__init__()

        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, zs, logdet, bits_per_pixel=False):
        log_prob = sum(self.base_dist.log_prob(z).sum([1, 2, 3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return log_prob


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


if __name__ == '__main__':
    x = Variable(
        torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).view(1, 2, 3, 3),
        requires_grad=True)
    addition = TVLoss()
    z = addition(x)
    print(x)
    print(z.data)
    z.backward()
    print(x.grad)
