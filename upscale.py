
import math
import numpy as np
import torch
import rrdbnet

class Upscaler(object):
    def upscale(self, input_image):
        # nop
        return input_image

class RRDBNetUpscaler(Upscaler):
    def __init__(self, model_data, device):
        try:
            # get largest model index from keys like "model.X.weight"
            max_index = max([int(n.split(".")[1]) for n in model_data.keys()])
        except:
            # invalid model dict format?
            raise RuntimeError("Unable to determine scale factor for model")

        # calculate scale factor from index
        # (1x=4, 2x=7, 4x=10, 8x=13, etc.)
        scale_factor = pow(2, (max_index - 4) // 3)

        model = rrdbnet.RRDB_Net(3, 3, 64, 23, upscale=scale_factor)
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

class TiledUpscaler(Upscaler):
    def __init__(self, upscaler, tile_size, tile_padding):
        self.upscaler = upscaler
        self.tile_size = tile_size
        self.tile_padding = tile_padding

    def upscale(self, input_image):
        scale_factor = self.upscaler.scale_factor
        width, height, depth = input_image.shape
        output_width = width * scale_factor
        output_height = height * scale_factor
        output_shape = (output_width, output_height, depth)

        # start with black image
        output_image = np.zeros(output_shape, np.uint8)

        tile_padding = math.ceil(self.tile_size * self.tile_padding)
        tile_size = math.ceil(self.tile_size / scale_factor)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size

                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)

                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_padding, 0)
                input_end_x_pad = min(input_end_x + tile_padding, width)

                input_start_y_pad = max(input_start_y - tile_padding, 0)
                input_end_y_pad = min(input_end_y + tile_padding, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                tile_idx = y * tiles_x + x + 1

                print("  Tile %d/%d (x=%d y=%d %dx%d)" % \
                    (tile_idx, tiles_x * tiles_y, x, y, input_tile_width, input_tile_height))

                input_tile = input_image[input_start_x_pad:input_end_x_pad, input_start_y_pad:input_end_y_pad]

                # upscale tile
                output_tile = self.upscaler.upscale(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * scale_factor
                output_end_x = input_end_x * scale_factor

                output_start_y = input_start_y * scale_factor
                output_end_y = input_end_y * scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * scale_factor

                output_start_y_tile = (input_start_y - input_start_y_pad) * scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * scale_factor

                # put tile into output image
                output_image[output_start_x:output_end_x, output_start_y:output_end_y] = \
                    output_tile[output_start_x_tile:output_end_x_tile, output_start_y_tile:output_end_y_tile]

        return output_image
