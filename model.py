import os
import collections
import torch

class Model(object):
    def name(self):
        pass

    def load(self):
        pass

class FileModel(Model):
    def __init__(self, path):
        self._model = None
        self._path = path

    def _get_scale_index(self, state_dict):
        # this is more or less guesswork, since I haven't seen any
        # non-4x models using the new format in the wild, but it
        # should work in theory
        max_index = 0

        for k in state_dict.keys():
            if k.startswith("upconv") and k.endswith(".weight"):
                max_index = max(max_index, int(k[6:-7]))

        return max_index

    def _get_legacy_scale_index(self, state_dict):
        try:
            # get largest model index from keys like "model.X.weight"
            max_index = max([int(n.split(".")[1]) for n in state_dict.keys()])
        except:
            # invalid model dict format?
            raise RuntimeError("Unable to determine scale index for model")

        return (max_index - 4) // 3

    def _build_legacy_keymap(self, n_upscale):
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

    def name(self):
        return os.path.splitext(os.path.basename(self._path))[0]

    def _load(self):
        state_dict = torch.load(self._path)

        # check for legacy model format
        if "model.0.weight" in state_dict:
            # remap dict keys to new format
            scale_index = self._get_legacy_scale_index(state_dict)
            keymap = self._build_legacy_keymap(scale_index)
            state_dict = {keymap[k]: v for k, v in state_dict.items()}
        else:
            scale_index = self._get_scale_index(state_dict)

        return state_dict, scale_index

    def load(self):
        if self._model is None:
            self._model = self._load()
        return self._model

class WeightedFileListModel(Model):
    def __init__(self, weight_map):
        self._models = {}
        self._total_weigth = 0

        names = []
        for path, weight in weight_map.items():
            model = FileModel(path)
            self._models[model] = weight

            names.append(model.name())
            names.append(str(weight))

        self._name = "_".join(names)

    def name(self):
        return self._name

    def load(self):
        net_interp = collections.OrderedDict()
        total_weigth = sum(self._models.values())
        scale = 0

        for model, weight in self._models.items():
            alpha = weight / total_weigth
            net, scale = model.load()
            for k, v in net.items():
                va = alpha * v
                if k in net_interp:
                    net_interp[k] += va
                else:
                    net_interp[k] = va

        return net_interp, scale
