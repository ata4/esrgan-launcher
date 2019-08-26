#!/usr/bin/python3
import sys
import os
import glob
import math
import argparse
import cv2
import numpy as np
import torch

import architecture as arch

class ESRGANUpscaler(object):
    def __init__(self, model_path, device, scale_factor):
        model_data = torch.load(model_path)

        model = arch.RRDB_Net(3, 3, 64, 23, upscale=scale_factor)
        model.load_state_dict(model_data, strict=True)
        model.eval()

        for _, v in model.named_parameters():
            v.requires_grad = False

        self.model = model.to(device)
        self.device = device
        self.scale_factor = scale_factor

    def upscale(self, input):
        input = input * 1.0 / 255
        input = np.transpose(input[:, :, [2, 1, 0]], (2, 0, 1))
        input = torch.from_numpy(input).float()
        input = input.unsqueeze(0).to(self.device)

        output = self.model(input).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        return output

class CubicUpscaler(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def upscale(self, input):
        return cv2.resize(input, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)

class TiledUpscaler(object):
    def __init__(self, upscaler, tile_size):
        self.upscaler = upscaler
        self.scale_factor = upscaler.scale_factor
        self.tile_size = tile_size

    def upscale(self, input):
        width, height, depth = input.shape
        output_shape = (width * self.scale_factor, height * self.scale_factor, depth)

        output = np.zeros(output_shape, np.uint8)

        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size

                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)

                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                tile_idx = y * tiles_x + x + 1

                print('  Tile %d/%d (x=%d y=%d %dx%d)' % (tile_idx, tiles_x * tiles_y, x, y, input_tile_width, input_tile_height), flush=True)

                input_tile = input[input_start_x:input_end_x, input_start_y:input_end_y]

                # upscale tile
                output_tile = self.upscaler.upscale(input_tile)

                # put tile into output image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor

                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                output[output_start_x:output_end_x, output_start_y:output_end_y] = output_tile

        return output

class ESRGAN:
    def __init__(self, upscaler):
        self.upscaler = upscaler
        self.per_channel = False

    def process(self, input_path, output_path):
        input = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if input is None:
            print("Unsupported image format:", input_path)
            return

        if len(input.shape) == 2:
            # only one channel
            input_rgb = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
            output_rgb = self.upscaler.upscale(input_rgb)
            output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY)
        elif self.per_channel:
            output_shape = list(input.shape)
            output_shape[0] *= self.upscaler.scale_factor
            output_shape[1] *= self.upscaler.scale_factor

            # create black image in upscaled resolution
            output = np.zeros(output_shape, np.uint8)

            # process every input channel individually as grayscale
            # and place it on the output channel
            for c in range(output_shape[2]):
                print(" Channel %d" % c)
                input_bw = input[:, :, c]
                input_rgb = cv2.cvtColor(input_bw, cv2.COLOR_GRAY2RGB)
                output_rgb = self.upscaler.upscale(input_rgb)
                output[:, :, c] = output_rgb[:, :, 0]

        else:
            # extract alpha channel if present
            if input.shape[2] == 4:
                input_alpha = input[:, :, 3]
                input = input[:, :, 0:3]
            else:
                input_alpha = None

            output = self.upscaler.upscale(input)

            # upscale alpha channel separately
            if input_alpha is not None:
                print(" Alpha")
                output = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA)
                input_alpha_rgb = cv2.cvtColor(input_alpha, cv2.COLOR_GRAY2RGB)
                output_alpha_rgb = self.upscaler.upscale(input_alpha_rgb)
                output[:, :, 3] = output_alpha_rgb[:, :, 0]

        cv2.imwrite(output_path, output)

def main():
    parser = argparse.ArgumentParser(description='ESRGAN image upscaler with tiling support')

    parser.add_argument('input', help='Path to input folder')
    parser.add_argument('output', help='Path to output folder')
    parser.add_argument('model', help='Path to model file')

    parser.add_argument('--tilesize', type=int, metavar='N', default=256, help='size of tiles in pixels (0 = don\'t use tiles)')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU/CUDA (very slow!)')
    parser.add_argument('--scale', type=int, metavar='S', default=4, help='scale factor of the output images')

    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    for current_model_path in glob.glob(args.model):
        if os.path.isdir(current_model_path):
            continue
            
        model_name = os.path.splitext(os.path.basename(current_model_path))[0]
        print("Initializing ESRGAN using model '%s'" % model_name, flush=True)

        output_dir = os.path.join(args.output, model_name)
        os.makedirs(output_dir, exist_ok=True)

        upscaler = ESRGANUpscaler(current_model_path, device, scale_factor=args.scale)
        upscaler = TiledUpscaler(upscaler, 512)
        esrgan = ESRGAN(upscaler)

        if os.path.isdir(args.input):
            for dirpath, _, filenames in os.walk(args.input):
                for filename in filenames:
                    input_path = os.path.join(dirpath, filename)
                    input_name = os.path.basename(input_path)
                    print('Processing', input_name, flush=True)

                    input_path_rel = os.path.relpath(input_path, args.input)
                    output_path_rel = os.path.splitext(input_path_rel)[0] + '.png'
                    output_path = os.path.join(output_dir, output_path_rel)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    esrgan.process(input_path, output_path)

        else:
            for input_path in glob.glob(args.input):
                if os.path.isdir(input_path):
                    continue

                input_name = os.path.basename(input_path)
                print('Processing', input_name, flush=True)

                output_name = os.path.splitext(input_name)[0] + '.png'
                output_path = os.path.join(output_dir, output_name)
                esrgan.process(input_path, output_path)

if __name__ == '__main__':
    exit(main())