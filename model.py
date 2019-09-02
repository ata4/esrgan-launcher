import os
import collections
import torch

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
