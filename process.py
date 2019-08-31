#!/usr/bin/python3
import sys
import os
import glob
import math
import argparse
import collections
import cv2
import numpy as np
import torch

import architecture as arch

class Upscaler(object):
    def __init__(self, model_data, device):
        # crude method to detect the scale factor for a model, but
        # it seems to work
        scale_factor = None
        for i in range(6):
            attr_key = "model.%d.weight" % (4 + 3 * i)
            if attr_key in model_data:
                scale_factor = 2 ** i
                break

        if scale_factor is None:
            # ... or maybe not
            raise RuntimeError("Unable to determine scale factor for model")

        model = arch.RRDB_Net(3, 3, 64, 23, upscale=scale_factor)
        model.load_state_dict(model_data, strict=True)
        model.eval()

        for _, v in model.named_parameters():
            v.requires_grad = False

        self.model = model.to(device)
        self.device = device
        self.scale_factor = scale_factor

    def upscale(self, input_image):
        input_image = input_image * 1.0 / 255
        input_image = np.transpose(input_image[:, :, [2, 1, 0]], (2, 0, 1))
        input_image = torch.from_numpy(input_image).float()
        input_image = input_image.unsqueeze(0).to(self.device)

        output_image = self.model(input_image).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_image = np.transpose(output_image[[2, 1, 0], :, :], (1, 2, 0))
        output_image = (output_image * 255.0).round()

        return output_image

class TiledUpscaler(object):
    def __init__(self, upscaler, tile_size):
        self.upscaler = upscaler
        self.scale_factor = upscaler.scale_factor
        self.tile_size = tile_size

    def upscale(self, input_image):
        width, height, depth = input_image.shape
        output_width = width * self.scale_factor
        output_height = height * self.scale_factor
        output_shape = (output_width, output_height, depth)

        # start with black image
        output_image = np.zeros(output_shape, np.uint8)

        tile_size = math.ceil(self.tile_size / self.scale_factor)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size

                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)

                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                tile_idx = y * tiles_x + x + 1

                print("  Tile %d/%d (x=%d y=%d %dx%d)" % (tile_idx, tiles_x * tiles_y, x, y, input_tile_width, input_tile_height))

                input_tile = input_image[input_start_x:input_end_x, input_start_y:input_end_y]

                # upscale tile
                output_tile = self.upscaler.upscale(input_tile)

                # put tile into output image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor

                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                output_image[output_start_x:output_end_x, output_start_y:output_end_y] = output_tile

        return output_image

class FileModel(object):
    def __init__(self, path):
        self._model = None
        self._path = path
        self._name = os.path.splitext(os.path.basename(path))[0]

    def name(self):
        return self._name

    def load(self):
        return torch.load(self._path)

    def get(self):
        if self._model is None:
            self._model = self.load()
        return self._model

class WeightedFileListModel(FileModel):
    def __init__(self, weight_map):
        self._model = None
        self._weight_map = weight_map

        names = []
        for path, weight in self._weight_map.items():
            names.append(os.path.splitext(os.path.basename(path))[0])
            names.append(str(weight))

        self._name = "_".join(names)

    def load(self):
        net_interp = collections.OrderedDict()
        total_weigth = sum(self._weight_map.values())

        for path, weight in self._weight_map.items():
            alpha = weight / total_weigth
            net = torch.load(path)
            for k, v in net.items():
                va = alpha * v
                if k in net_interp:
                    net_interp[k] += va
                else:
                    net_interp[k] = va

        return net_interp

class ESRGAN(object):
    def __init__(self):
        self.per_channel = False
        self.device = "cpu"
        self.torch = None
        self.tilesize = 512

        self.models_upscale = []
        self.models_filter = []

    def _process_image(self, input_image, upscaler):
        if self.tilesize > 0:
            upscaler = TiledUpscaler(upscaler, self.tilesize)

        if len(input_image.shape) == 2:
            # only one channel
            input_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            output_rgb = upscaler.upscale(input_rgb)
            output_image = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY)
        elif self.per_channel:
            output_shape = list(input_image.shape)
            output_shape[0] *= upscaler.scale_factor
            output_shape[1] *= upscaler.scale_factor

            # create black image in upscaled resolution
            output_image = np.zeros(output_shape, np.uint8)

            # process every input channel individually as grayscale
            # and place it on the output channel
            for c in range(output_shape[2]):
                print(" Channel %d" % c)
                input_bw = input_image[:, :, c]
                input_rgb = cv2.cvtColor(input_bw, cv2.COLOR_GRAY2RGB)
                output_rgb = upscaler.upscale(input_rgb)
                output_image[:, :, c] = output_rgb[:, :, 0]

        else:
            # extract alpha channel if present
            if input_image.shape[2] == 4:
                input_alpha = input_image[:, :, 3]
                input_image = input_image[:, :, 0:3]
            else:
                input_alpha = None

            output_image = upscaler.upscale(input_image)

            # upscale alpha channel separately
            if input_alpha is not None:
                print(" Alpha")
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2RGBA)
                input_alpha_rgb = cv2.cvtColor(input_alpha, cv2.COLOR_GRAY2RGB)
                output_alpha_rgb = upscaler.upscale(input_alpha_rgb)
                output_image[:, :, 3] = output_alpha_rgb[:, :, 0]

        return output_image

    def _process_file(self, input_path, output_path, models):
        input_name = os.path.basename(input_path)
        print("Processing", input_name)

        # read image
        input_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if input_image is None:
            print("Unsupported image format:", input_path)
            return

        output_image = input_image

        for model in models:
            print("Applying model '%s'" % model.name())
            upscaler = Upscaler(model.get(), self.torch)
            output_image = self._process_image(output_image, upscaler)

        # write image
        cv2.imwrite(output_path, output_image)

    def _parse_model(self, model_args):
        models = []

        if not model_args:
            return models

        for model_path in model_args:
            # if the model path contains path separators, it's a weighted
            # model that contains a list of model files and their weight
            if os.path.pathsep in model_path:
                path = None
                model_paths = {}
                model_parts = model_path.split(os.path.pathsep)

                if len(model_parts) % 2 != 0:
                    raise RuntimeError("Model list must be in format [model path]%c[weight]%c..." % (os.path.pathsep, os.path.pathsep))

                for model_part in model_parts:
                    if path is None:
                        path = model_part
                    else:
                        try:
                            weight = int(model_part)
                        except ValueError:
                            raise RuntimeError("Invalid number for model weight:", model_part)
                        model_paths[path] = weight
                        path = None

                models.append(WeightedFileListModel(model_paths))
            else:
                for current_model_path in glob.glob(model_path):
                    if os.path.isdir(current_model_path):
                        continue

                    models.append(FileModel(current_model_path))

        return models

    def main(self):
        parser = argparse.ArgumentParser(description="ESRGAN image upscaler")

        parser.add_argument("input", help="Path to input folder")
        parser.add_argument("output", help="Path to output folder")

        parser.add_argument("--model", action="append", help="path to upscaling model file (can be used repeatedly)")
        parser.add_argument("--filter", action="append", help="path to 1x filter model file (can be used repeatedly)")
        parser.add_argument("--tilesize", type=int, metavar="N", default=self.tilesize, help="width/height of tiles in pixels (0 = don't use tiles)")
        parser.add_argument("--device", default=self.device, help="use this Torch device (typically 'cpu' or 'cuda')")

        args = parser.parse_args()

        self.device = args.device
        self.torch = torch.device(self.device)
        self.tilesize = args.tilesize

        self.models_upscale = self._parse_model(args.model)
        self.models_filter = self._parse_model(args.filter)

        if not self.models_upscale:
            print("No model selected, use '--model' to choose at least one model")
            return 1

        for model_upscale in self.models_upscale:
            models = list(self.models_filter)
            models.append(model_upscale)

            model_name = "_".join([model.name() for model in models])

            print("Model pass '%s'" % model_name)

            output_dir = os.path.join(args.output, model_name)
            os.makedirs(output_dir, exist_ok=True)

            if os.path.isdir(args.input):
                for dirpath, _, filenames in os.walk(args.input):
                    for filename in filenames:
                        input_path = os.path.join(dirpath, filename)

                        input_path_rel = os.path.relpath(input_path, args.input)
                        output_path_rel = os.path.splitext(input_path_rel)[0] + ".png"
                        output_path = os.path.join(output_dir, output_path_rel)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        self._process_file(input_path, output_path, models)

            else:
                for input_path in glob.glob(args.input):
                    if os.path.isdir(input_path):
                        continue

                    input_name = os.path.basename(input_path)
                    output_name = os.path.splitext(input_name)[0] + ".png"
                    output_path = os.path.join(output_dir, output_name)
                    self._process_file(input_path, output_path, models)

if __name__ == "__main__":
    exit(ESRGAN().main())
