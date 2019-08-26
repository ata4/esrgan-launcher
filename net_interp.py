import sys
import torch
from collections import OrderedDict

net_path_1 = sys.argv[1]
net_path_2 = sys.argv[2]
net_interp_path = sys.argv[3]

alpha = float(sys.argv[4])

net_1 = torch.load(net_path_1)
net_2 = torch.load(net_path_2)
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_1.items():
    v_ESRGAN = net_2[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)
