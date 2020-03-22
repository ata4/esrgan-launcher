#!/usr/bin/python3
import sys
import os
import glob
import argparse
import cv2
import numpy as np
import torch
import model
import upscale

class ESRGAN(object):
    def __init__(self):
        self.device = ("cpu", "cuda")[torch.cuda.is_available()]
        self.torch = None
        self.tile_size = 512
        self.tile_padding = 0.125
        self.per_channel = False
        self.no_alpha = False

        self.models_upscale = []
        self.models_prefilter = []
        self.models_postfilter = []

    def _process_image(self, input_image, upscaler):
        if self.tile_size > 0:
            upscaler = upscale.TiledUpscaler(upscaler, self.tile_size, self.tile_padding)

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
                if self.no_alpha:
                    input_alpha = None
                else:
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

        # read input
        input_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if input_image is None:
            print("Unsupported image format:", input_path)
            return

        # start with an unmodified image
        output_image = input_image

        # apply all models from the list
        for model in models:
            print("Applying model '%s'" % model.name())
            upscaler = upscale.RRDBNetUpscaler(model, self.torch)
            output_image = self._process_image(output_image, upscaler)

        # write output
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

                # it's a list of pairs, so the number of elements must be even
                if len(model_parts) % 2 != 0:
                    raise RuntimeError("Model list must be in format [model path]%c[weight]%c..." % (os.path.pathsep, os.path.pathsep))

                for model_part in model_parts:
                    if path is None:
                        # start with the model path first
                        path = model_part
                    else:
                        # then parse the model weight
                        try:
                            weight = int(model_part)
                        except ValueError:
                            raise RuntimeError("Invalid number for model weight:", model_part)

                        # and put it into the map
                        model_paths[path] = weight

                        # continue with next pair
                        path = None

                models.append(model.WeightedFileListModel(model_paths))
            else:
                # simple path, may use wildcards
                for current_model_path in glob.glob(model_path):
                    if os.path.isdir(current_model_path):
                        continue

                    models.append(model.FileModel(current_model_path))

        return models

    def main(self):
        parser = argparse.ArgumentParser(description="ESRGAN image upscaler")

        parser.add_argument("input", help="Path to input folder")
        parser.add_argument("output", help="Path to output folder")

        parser.add_argument("--model", action="append", required=True, help="path to upscaling model file (can be used repeatedly)")
        parser.add_argument("--device", default=self.device, help="select Torch device (typically 'cpu' or 'cuda')")
        parser.add_argument("--prefilter", action="append", metavar="FILE", help="path to model file applied before upscaling (can be used repeatedly)")
        parser.add_argument("--postfilter", action="append", metavar="FILE", help="path to model file applied after upscaling (can be used repeatedly)")
        parser.add_argument("--tilesize", type=int, metavar="N", default=self.tile_size, help="width/height of tiles in pixels (0 = don't use tiles)")
        parser.add_argument("--perchannel", action="store_true", help="process each channel individually as grayscale image")
        parser.add_argument("--noalpha", action="store_true", help="ignore alpha channels from input and output RGB only")

        args = parser.parse_args()

        self.device = args.device
        self.torch = torch.device(self.device)
        self.tile_size = args.tilesize
        self.per_channel = args.perchannel
        self.no_alpha = args.noalpha

        self.models_upscale = self._parse_model(args.model)
        self.models_prefilter = self._parse_model(args.prefilter)
        self.models_postfilter = self._parse_model(args.postfilter)

        if not any((self.models_upscale, self.models_prefilter, self.models_postfilter)):
            print("No models selected or found!")
            return 1

        for model_upscale in self.models_upscale:
            models = self.models_prefilter + [model_upscale] + self.models_postfilter
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
