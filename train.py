# -*-coding:utf-8-*-
import os
import time
import datetime
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from train_utils.utils import create_folders, weight_init, save_fixedweight, get_params_groups, CombinedLR, time_synchronized
from train_utils.dataset import ICCADDataset
from src.MFRLNet import MFRLNet
from train_utils.loss import Loss
from train_utils.metric import Acc_FA
import random
from thop import profile


class Trainer(object):
    def __init__(self, args, folder, folder_name):
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        # dataset
        self.trainset = ICCADDataset(args, mode="train", case=args.case, ratio=args.ratio, firstepochs=args.firstepochs)
        self.valset = ICCADDataset(args, mode="test", case=args.case)
        num_workers = min([os.cpu_count(), args.batch_size])
        self.train_data_loader = Data.DataLoader(self.trainset,
                                                 batch_size=args.batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True,
                                                 pin_memory=True)
        self.val_data_loader = Data.DataLoader(self.valset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True)

        self.model = MFRLNet()
        weight_init(self.model)
        # checkpoint = torch.load("./model_fixed.pth", map_location="cpu")
        # self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        save_fixedweight(self.model, folder)

        # calculate Params Flops
        # flops, params = profile(self.model, inputs=(torch.randn(1, 3, args.base_size, args.base_size, device=self.device), ))
        # print(f"Params: {params / 1e6:.2f}M")
        # print(f"Flops: {flops / 1e9:.2f}G")

        # optimizer
        params_group = get_params_groups(self.model, weight_decay=args.weight_decay)
        self.optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CombinedLR(optimizer=self.optimizer, firstepochs=args.firstepochs)
        # evaluation metrics
        self.Acc_FA = Acc_FA()

        self.best_Acc = 0.0

        # SummaryWriter
        self.writer = SummaryWriter(log_dir=folder, flush_secs=5)
        self.writer.add_text(folder_name, "args:%s," % args)

        self.loss, self.lr = 0.0, 0.0

    def training(self, epoch):
        self.trainset.dataset_update(epoch)
        self.model.train()
        self.Acc_FA.reset()
        losses = []
        tbar = tqdm(self.train_data_loader)
        for image, mask, label in tbar:
            image, mask, label = image.to(self.device), mask.to(self.device), label.to(self.device)
            seg, output = self.model(image)
            loss = Loss(seg, mask, output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.Acc_FA.update(output.detach(), label)
            Acc, FA = self.Acc_FA.get()

            self.lr = trainer.optimizer.param_groups[0]["lr"]
            losses.append(loss.item())
            self.loss = np.mean(losses)
            tbar.set_description("Epoch: [%d] lr: %.6f train_loss: %.6f train_Acc: %.3f train_FA: %.3f" % (epoch, self.lr, self.loss, Acc, FA))

        self.scheduler.step(epoch)

        self.writer.add_scalar("train_loss", self.loss, epoch)
        self.writer.add_scalar("lr", self.lr, epoch)
        self.writer.add_scalar("train_Acc", Acc, epoch)
        self.writer.add_scalar("train_FA", FA, epoch)

    def validation(self, epoch):
        self.model.eval()
        self.Acc_FA.reset()
        itimes = []
        eval_losses = []
        tbar = tqdm(self.val_data_loader)
        with torch.no_grad():
            for image, mask, label in tbar:
                image, mask, label = image.to(self.device), mask.to(self.device), label.to(self.device)

                t_start = time_synchronized(self.device)
                seg, output = self.model(image)
                t_end = time_synchronized(self.device)
                itimes.append((t_end - t_start) / image.shape[0])

                loss = Loss(seg, mask, output, label)
                eval_losses.append(loss.item())
                self.Acc_FA.update(output, label)
                Acc, FA = self.Acc_FA.get()
                tbar.set_description("Val: val_loss: %.6f itime: %.7f eval_Acc: %.3f eval_FA: %.3f" % (np.mean(eval_losses), np.mean(itimes), Acc, FA))

        pth_name = "Epoch-%d_Acc-%.3f_FA-%.3f.pth" % (epoch, Acc, FA)
        save_file = self.model.state_dict()

        self.writer.add_scalar("eval_loss", np.mean(eval_losses), epoch)
        self.writer.add_scalar("eval_Acc", Acc, epoch)
        self.writer.add_scalar("eval_FA", FA, epoch)

        results_file = os.path.join(folder, f"results{folder_name}.txt")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、val_loss、lr以及验证集各指标
            write_info = f"[epoch: {epoch}] lr: {self.lr:.6f} train_loss: {self.loss:.6f} val_loss: {np.mean(eval_losses):.6f} itime: {np.mean(itimes):.7f} \n " \
                         f"\t\t\tAcc: {Acc:.3f} FA: {FA:.3f} \n\n" \

            f.write(write_info)

        # save_best
        # 避免类别不平衡造成Acc过高，且F1过低，FA应当做如下控制（仅为自动选择最优结果，不影响模型训练和验证）：
        # ICCAD19-1: FA < 3.0; ICCAD19-2: FA < 85.0
        # Note：本工程并未集成F1实现，可在train_utils/metric.py/Acc_FA中参考论文关于F1定义自行实现，并将此处条件控制修改为“Acc > self.best_Acc and F1 > self.best_F1”以便实现更平衡的结果选择
        if Acc > self.best_Acc and FA < 3.0:
            torch.save(save_file, os.path.join(folder, "best_weights", pth_name))
            self.best_Acc = Acc

        # only save latest args.epochs(default: 15) epoch weights
        # if os.path.exists(f"{folder}/save_weights/model_{epoch - args.epochs}.pth"):
        #     os.remove(f"{folder}/save_weights/model_{epoch - args.epochs}.pth")
        #
        # torch.save(save_file, f"{folder}/save_weights/model_{epoch}.pth")


def parse_args():
    import argparse
    # Setting parameters
    parser = argparse.ArgumentParser(description="Implement of lnet model")

    parser.add_argument("--data-path", type=str, default="E:/LHT/datasets/classify/images/ICCAD19", help="dataset root")
    parser.add_argument("--case", type=str, default="1", help="dataset case")
    parser.add_argument("--base-size", type=int, default=256, help="base image size")

    # Training parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="training device")
    parser.add_argument("--batch-size", type=int, default=96, help="batch_size for training")
    parser.add_argument("--ratio", type=int, default=0.1, help="sampling ratio of NHS number during training")
    parser.add_argument("--epochs", type=int, default=900, help="number of epochs")
    parser.add_argument("--firstepochs", type=int, default=60, help="epochs of the first stage of training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0, help="weight decay")
    parser.add_argument("--deterministic", type=bool, default=False, help="deterministic mode")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # set random seed
    seed = 416
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = parse_args()

    # set deterministic mode
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

    folder, folder_name = create_folders()

    trainer = Trainer(args, folder, folder_name)
    start_time = time.time()

    for epoch in range(args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time:", total_time_str)
