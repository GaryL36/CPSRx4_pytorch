import torch
import numpy as np

def coefficient_decompose(coefficient):
    if type(coefficient) == np.ndarray:
        c1_dec = coefficient[0][0]
        c2_dec = coefficient[0][1]
        c3_dec = coefficient[0][2]
        c4_dec = coefficient[0][3]
        c5_dec = coefficient[0][4]
        c6_dec = coefficient[0][5]

        c1_tensor = torch.FloatTensor(c1_dec[0][0]).unsqueeze(0)

        c2_tensor = torch.FloatTensor(c2_dec[0][0]).unsqueeze(0)
        index_2 = []    # transpose index
        for d in range(1, c2_dec[0].size):
            c2_dec[0][d] = torch.FloatTensor(c2_dec[0][d])
            if c2_dec[0][d].size(0) != c2_tensor.size(1):
                c2_dec[0][d] = c2_dec[0][d].transpose(0, 1)
                index_2.append(d)
            c2_tensor = torch.cat([c2_dec[0][d].unsqueeze(0), c2_tensor], dim=0)

        c3_tensor = torch.FloatTensor(c3_dec[0][0]).unsqueeze(0)
        index_3 = []
        for d in range(1, c3_dec[0].size):
            c3_dec[0][d] = torch.FloatTensor(c3_dec[0][d])
            if c3_dec[0][d].size(0) != c3_tensor.size(1):
                c3_dec[0][d] = c3_dec[0][d].transpose(0, 1)
                index_3.append(d)
            c3_tensor = torch.cat([c3_dec[0][d].unsqueeze(0), c3_tensor], dim=0)

        c4_tensor = torch.FloatTensor(c4_dec[0][0]).unsqueeze(0)
        index_4 = []
        for d in range(1, c4_dec[0].size):
            c4_dec[0][d] = torch.FloatTensor(c4_dec[0][d])
            if c4_dec[0][d].size(0) != c4_tensor.size(1):
                c4_dec[0][d] = c4_dec[0][d].transpose(0, 1)
                index_4.append(d)
            c4_tensor = torch.cat([c4_dec[0][d].unsqueeze(0), c4_tensor], dim=0)

        c5_tensor = torch.FloatTensor(c5_dec[0][0]).unsqueeze(0)
        index_5 = []
        for d in range(1, c5_dec[0].size):
            c5_dec[0][d] = torch.FloatTensor(c5_dec[0][d])
            if c5_dec[0][d].size(0) != c5_tensor.size(1):
                c5_dec[0][d] = c5_dec[0][d].transpose(0, 1)
                index_5.append(d)
            c5_tensor = torch.cat([c5_dec[0][d].unsqueeze(0), c5_tensor], dim=0)

        c6_tensor = torch.FloatTensor(c6_dec[0][0]).unsqueeze(0)

        return c1_tensor, c2_tensor, c3_tensor, c4_tensor, c5_tensor, c6_tensor, index_2, index_3, index_4, index_5
    else:
        raise Exception('Unknown Type', type(coefficient))



def coefficient_recompose(c1, c2, c3, c4, c5, c6, i2, i3, i4, i5, coefficient):
    if type(coefficient) == np.ndarray:
        c_rec = coefficient.copy()
        c1_rec = coefficient[0][0].copy()
        c2_rec = coefficient[0][1].copy()
        c3_rec = coefficient[0][2].copy()
        c4_rec = coefficient[0][3].copy()
        c5_rec = coefficient[0][4].copy()
        c6_rec = coefficient[0][5].copy()

        c1_rec[0][0] = c1[0].numpy()

        for d in range(0, c2.size(0)):
            transpose_flag = 0
            for i in range(0, len(i2)):
                if d == i2[i]:
                    transpose_flag = 1
                    break
            if transpose_flag == 1:
                c2_rec[0][d] = c2[d].transpose(0, 1).numpy()
            else:
                c2_rec[0][d] = c2[d].numpy()

        for d in range(0, c3.size(0)):
            transpose_flag = 0
            for i in range(0, len(i3)):
                if d == i3[i]:
                    transpose_flag = 1
                    break
            if transpose_flag == 1:
                c3_rec[0][d] = c3[d].transpose(0, 1).numpy()
            else:
                c3_rec[0][d] = c3[d].numpy()

        for d in range(0, c4.size(0)):
            transpose_flag = 0
            for i in range(0, len(i3)):
                if d == i4[i]:
                    transpose_flag = 1
                    break
            if transpose_flag == 1:
                c4_rec[0][d] = c4[d].transpose(0, 1).numpy()
            else:
                c4_rec[0][d] = c4[d].numpy()

        for d in range(0, c5.size(0)):
            transpose_flag = 0
            for i in range(0, len(i3)):
                if d == i5[i]:
                    transpose_flag = 1
                    break
            if transpose_flag == 1:
                c5_rec[0][d] = c5[d].transpose(0, 1).numpy()
            else:
                c5_rec[0][d] = c5[d].numpy()

        c6_rec[0][0] = c6[0].numpy()

        c_rec[0][0] = c1_rec
        c_rec[0][1] = c2_rec
        c_rec[0][2] = c3_rec
        c_rec[0][3] = c4_rec
        c_rec[0][4] = c5_rec
        c_rec[0][5] = c6_rec

        return c_rec
    else:
        raise Exception('Unknown Type', type(coefficient))



def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        # img(H,W,C)
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.

        # 第0维度——>第2维度，第1维度——>第0维度，第2维度——>第1维度
        # (C,W,H)——>(W,H,C)
        # 由[y.(x,y), cb.(x,y), cr.(x,y)]变为[(x,y).y, (x,y).cb, (x,y).cr]
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))

# train时候图片像素取值[0.，1.]
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

# test时候图片像素取值[0.，255.]
def calc_psnr_255test(img1, img2):
    return 10. * torch.log10(255. / torch.mean((img1 - img2) ** 2))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count