import os
import time, datetime
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def create_folders(folder="result"):
    folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f"./{folder}"):
        os.makedirs(f"./{folder}")
    if not os.path.exists(f"./{folder}/{folder_name}"):
        os.makedirs(f"./{folder}/{folder_name}")
    if not os.path.exists(f"./{folder}/{folder_name}/fixed_weight"):
        os.makedirs(f"./{folder}/{folder_name}/fixed_weight")
    if not os.path.exists(f"./{folder}/{folder_name}/save_weights"):
        os.makedirs(f"./{folder}/{folder_name}/save_weights")
    if not os.path.exists(f"./{folder}/{folder_name}/best_weights"):
        os.makedirs(f"./{folder}/{folder_name}/best_weights")

    return os.path.join(folder, folder_name), folder_name


def weight_init(model):
    for m, _ in model.named_parameters():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# 保存每次训练时随机加载的网络结构权重，保证可以重新训练复现原先训练效果
def save_fixedweight(model, folder):
    save_file = model.state_dict()
    torch.save(save_file, f"./{folder}/fixed_weight/model_fixed.pth")


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group


class CombinedLR():
    def __init__(self, optimizer, firstepochs=60):
        super().__init__()
        self.firstepochs = firstepochs
        self.steplr_scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        self.cosinelr_scheduler = CosineAnnealingLR(optimizer, T_max=30)

    def step(self, epoch):
        if epoch < self.firstepochs:
            self.steplr_scheduler.step()
        else:
            self.cosinelr_scheduler.step()


def time_synchronized(device):
    torch.cuda.synchronize(device) if torch.cuda.is_available() and device.type != "cpu" else None

    return time.time()
