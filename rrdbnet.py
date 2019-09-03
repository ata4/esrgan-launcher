import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def get_scale_index(state_dict):
    # this is more or less guesswork, since I haven't seen any
    # non-4x models using the new format in the wild, but it
    # should work in theory
    max_index = 0

    for k in state_dict.keys():
        if k.startswith("upconv") and k.endswith(".weight"):
            max_index = max(max_index, int(k[6:-7]))

    return max_index

def get_legacy_scale_index(state_dict):
    try:
        # get largest model index from keys like "model.X.weight"
        max_index = max([int(n.split(".")[1]) for n in state_dict.keys()])
    except:
        # invalid model dict format?
        raise RuntimeError("Unable to determine scale index for model")

    return (max_index - 4) // 3

def build_legacy_keymap(n_upscale):
    keymap = collections.OrderedDict()
    keymap["model.0"] = "conv_first"

    for i in range(23):
        for j in range(1, 4):
            for k in range(1, 6):
                k1 = "model.1.sub.%d.RDB%d.conv%d.0" % (i, j, k)
                k2 = "RRDB_trunk.%d.RDB%d.conv%d" % (i, j, k)
                keymap[k1] = k2

    keymap["model.1.sub.23"] = "trunk_conv"

    n = 0
    for i in range(1, n_upscale + 1):
        n += 3
        k1 = "model.%d" % n
        k2 = "upconv%d" % i
        keymap[k1] = k2

    keymap["model.%d" % (n + 2)] = "HRconv"
    keymap["model.%d" % (n + 4)] = "conv_last"

    # add ".weigth" and ".bias" suffixes to all keys
    keymap_final = collections.OrderedDict()

    for k1, k2 in keymap.items():
        for k_type in ("weight", "bias"):
            k1_f = k1 + "." +  k_type
            k2_f = k2 + "." +  k_type
            keymap_final[k1_f] = k2_f

    return keymap_final

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.n_upscale = 0
        self.nf = nf

    def load_state_dict(self, state_dict, strict=True):
        # check for legacy model format
        if "model.0.weight" in state_dict:
            # remap dict keys to new format
            self.n_upscale = get_legacy_scale_index(state_dict)
            keymap = build_legacy_keymap(self.n_upscale)
            state_dict = {keymap[k]: v for k, v in state_dict.items()}
        else:
            self.n_upscale = get_scale_index(state_dict)

        # build upconv layers based on model scale
        for n in range(1, self.n_upscale + 1):
            upconv = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
            setattr(self, "upconv%d" % n, upconv)

        return super().load_state_dict(state_dict, strict)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        # apply upconv layers
        for n in range(1, self.n_upscale + 1):
            upconv = getattr(self, "upconv%d" % n)
            fea = self.lrelu(upconv(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
