import os
import json
import random
import collections
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def as_numpy(tensor, first_elem):
    if first_elem:
        return tensor.detach().cpu().numpy()[0, 0, ...]
    else:
        return tensor.detach().cpu().numpy()


def load_json(path):
    def _json_object_hook(d):
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def load_model(model, save_path):
    model_state = torch.load(save_path)
    if isinstance(model_state, nn.DataParallel):
        model_state = model_state.module.state_dict()
    else:
        model_state = model_state.state_dict()
    model.load_state_dict(model_state)


def norm(x):
    x = 2.0 * (x - 0.5)
    return x.clamp_(-1, 1)


def denorm(x):
    x = (x + 1) / 2.0
    return x.clamp_(0, 1)


def minmax_norm(x):
    vmax = np.max(x)
    vmin = np.min(x)
    x -= vmin
    x /= (vmax - vmin)
    return x


def minmax_norm(x):
    vmax = np.max(x)
    vmin = np.min(x)
    x -= vmin
    x /= (vmax - vmin)
    return x


def get_output_dir_path(config):
    study_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name = config.save.study_name + '_' + study_time
    output_dir_path = os.path.join(config.save.output_root_dir, dir_name)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )


class OneHotEncoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.ones = torch.sparse.torch.eye(n_classes).cuda()

    def forward(self, t):
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])
        t = t.data.long().contiguous().view(-1).cuda()
        out = Variable(self.ones.index_select(0, t)).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).float()
        return out
