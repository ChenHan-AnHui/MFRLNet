import torch


class Acc_FA():
    def __init__(self, score_thresh=0.35):
        super().__init__()
        self.score_thresh = score_thresh
        self.chs = 0
        self.cnhs = 0
        self.ahs = 0
        self.anhs = 0

    def update(self, inputs, labels):
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.score_thresh).type(torch.uint8)
        labels = (labels > self.score_thresh).type(torch.uint8)
        predicts = inputs + labels
        self.chs += torch.count_nonzero(predicts == 2)
        self.cnhs += torch.count_nonzero(predicts == 0)
        self.ahs += labels.sum()
        self.anhs += len(labels) - labels.sum()

    def get(self):
        Acc = self.chs / (self.ahs + 10e-9)
        FA = (self.anhs - self.cnhs) / (self.anhs + 10e-9)

        return Acc * 100, FA * 100

    def reset(self):
        self.chs = 0
        self.cnhs = 0
        self.ahs = 0
        self.anhs = 0
