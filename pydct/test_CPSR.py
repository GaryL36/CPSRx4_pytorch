import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from model import CPSR_s
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, coefficient_decompose, coefficient_recompose
import pyct as ct

if __name__ == '__main__':

    scale = 4
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = pil_image.open("/path/to/your/image").convert('RGB')
    bicubic = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
    bicubic = np.array(bicubic).astype(np.float32)
    ycbcr_bic = convert_rgb_to_ycbcr(bicubic)
    y_bic = ycbcr_bic[..., 0]

    # Curvelet Transform
    FDCT = ct.fdct2(y_bic.shape, 6, 16, True, cpx=True)
    y_coefficients = FDCT.fwd(y_bic)  # y_coefficients is a cell array

    c1_lr, c2_lr, c3_lr, c4_lr, c5_lr, c6_lr, i2, i3, i4, i5 = coefficient_decompose(y_coefficients)

    Net1 = CPSR_s(inC=1, outC=1).to(device)
    Net1.load_state_dict(torch.load("/path/to/x4/Net1_best.pth"))
    Net1.eval()
    c1_lr = c1_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        c1_sr = Net1(c1_lr).clamp(0.0, 1.0)
    c1_sr = c1_sr.squeeze(0).cpu()


    Net2 = CPSR_s(inC=16, outC=16).to(device)
    Net2.load_state_dict(torch.load("/path/to/x4/Net2_best.pth"))
    Net2.eval()
    c2_lr = c2_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        c2_sr = Net2(c2_lr).clamp(0.0, 1.0)
    c2_sr = c2_sr.squeeze(0).cpu()


    Net3 = CPSR_s(inC=32, outC=32).to(device)
    Net3.load_state_dict(torch.load("/path/to/x4/Net3_best.pth"))
    Net3.eval()
    c3_lr = c3_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        c3_sr = Net3(c3_lr).clamp(0.0, 1.0)
    c3_sr = c3_sr.squeeze(0).cpu()


    Net4 = CPSR_s(inC=32, outC=32).to(device)
    Net4.load_state_dict(torch.load("/path/to/x4/Net4_best.pth"))
    Net4.eval()
    c4_lr = c4_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        c4_sr = Net4(c4_lr).clamp(0.0, 1.0)
    c4_sr = c4_sr.squeeze(0).cpu()


    Net5 = CPSR_s(inC=64, outC=64).to(device)
    Net5.load_state_dict(torch.load("/path/to/x4/Net5_best.pth"))
    Net5.eval()
    c5_lr = c5_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        c5_sr = Net5(c5_lr).clamp(0.0, 1.0)
    c5_sr = c5_sr.squeeze(0).cpu()


    Net6 = CPSR_s(inC=1, outC=1).to(device)
    Net6.load_state_dict(torch.load("/path/to/x4/Net6_best.pth"))
    Net6.eval()
    c6_lr = c6_lr.unsqueeze(0).to(device)
    with torch.no_grad():
        c6_sr = Net6(c6_lr).clamp(0.0, 1.0)
    c6_sr = c6_sr.squeeze(0).cpu()

    y_recompose = coefficient_recompose(c1_sr, c2_sr, c3_sr, c4_sr, c5_sr, c6_sr, i2, i3, i4, i5, y_coefficients)

    # Inverse Curvelet Transform
    y_sr = FDCT.inv(y_recompose)

    output = np.array([y_sr, ycbcr_bic[..., 1], ycbcr_bic[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.show()
