import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class SKFFM(nn.Module):
    """Selective Kernel Feature Fusion Module."""

    def __init__(self, c, height=4, reduction=2):
        super().__init__()
        self.height = height
        d = max(int(c / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cv = nn.Sequential(nn.Conv2d(c, d, kernel_size=1), nn.ReLU())

        self.fcs = nn.ModuleList([])
        for _ in range(self.height):
            self.fcs.append(nn.Conv2d(d, c, kernel_size=1))

    def forward(self, x):
        b = x[0].shape[0]
        c = x[0].shape[1]

        x = torch.concat(x, 1)
        x = x.view(b, self.height, c, x.shape[2], x.shape[3])

        att = torch.sum(x, 1)
        att = self.avg_pool(att)
        att = self.cv(att)

        att = [fc(att) for fc in self.fcs]
        att = torch.concat(att, 1)
        att = att.view(b, self.height, c, 1, 1)

        att = torch.softmax(att, 1)

        x = torch.sum(x * att, 1)
        return x


class ACFFNet(nn.Module):
    """Adaptive Cross-layer Feature Fusion Network."""

    def __init__(self, channels=[16, 16, 32, 64]):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

        self.cv2 = Conv(c2, c1, 1, 1)
        self.cv3 = Conv(c3, c1, 1, 1)
        self.cv4 = Conv(c4, c1, 1, 1)
        self.skffm = SKFFM(c1)

    def forward(self, x):
        x1, x2, x3, x4 = x
        x2 = self.cv2(x2)
        x3 = self.cv3(x3)
        x4 = self.cv4(x4)
        x = self.skffm([x1, self.up2(x2), self.up4(x3), self.up8(x4)])
        return x
    

class Segment(nn.Module):
    """Segment Head."""

    def __init__(self, c1=16, c2=1, r=4):
        super().__init__()
        c_ = max(c1 // r, 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.cv = nn.Sequential(Conv(c1, c_, 3, 1),
                                nn.Conv2d(c_, c2, kernel_size=1))
        
    def forward(self, x):
        return self.up2(self.cv(x))
    
    
class MessageAgg(nn.Module):
    """Message Aggregation."""

    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HGConv(nn.Module):
    """Hypergraph Convolution."""

    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)
        return x


class HCM(nn.Module):
    """Hypergraph Compute Module."""

    def __init__(self, c, threshold=8):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HGConv(c, c)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg)
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x) + res)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation."""

    def __init__(self, c, r=4):
        super().__init__()
        c_ = max(c // r, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c_, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.fc(x)
        return x * att + x
    

class CA(nn.Module):
    """Coordinate Attention."""

    def __init__(self, c, r=4):
        super().__init__()
        c_ = max(c // r, 4)
        self.cv = nn.Conv1d(c, c_, kernel_size=3, padding=1)
        self.cv_h = nn.Conv1d(c_, c, kernel_size=3, padding=1)
        self.cv_v = nn.Conv1d(c_, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_h, x_v = self.relu(self.cv(torch.concat([x.mean(dim=3), x.mean(dim=2)], 2))).chunk(2, 2)
        x_h_att = torch.sigmoid(self.cv_h(x_h)).unsqueeze(3)
        x_v_att = torch.sigmoid(self.cv_v(x_v)).unsqueeze(2)
        att = torch.matmul(x_h_att, x_v_att)
        return att * x + x
    

class MFMI(nn.Module):
    """Multi-feature Modeling and Interaction."""

    def __init__(self, c1, c2, layer_num=4, r=4, hcm=False):
        super().__init__()
        c_ = max(c1 // r, 4)
        self.conv_reduction = Conv(c1, c_, 1, 1)

        if hcm:
            self.local_conv1 = nn.Sequential(Conv(c_, c_, 3, 1, g=c_), HCM(c_))
            self.local_conv2 = nn.Sequential(Conv(c_, c_, 3, 1, g=c_), HCM(c_))
            self.local_conv3 = nn.Sequential(Conv(c_, c_, 3, 1, g=c_), HCM(c_))
            self.local_conv4 = nn.Sequential(Conv(c_, c_, 3, 1, g=c_), HCM(c_))
        else:
            self.local_conv1 = Conv(c_, c_, 3, 1, g=c_)
            self.local_conv2 = Conv(c_, c_, 3, 1, g=c_)
            self.local_conv3 = Conv(c_, c_, 3, 1, g=c_)
            self.local_conv4 = Conv(c_, c_, 3, 1, g=c_)

        self.conv_fusion = Conv(c_ * layer_num, c2, 1, 1)
        self.se = SE(c2)
        self.ca = CA(c2)

    def forward(self, x):
        x_reduced = self.conv_reduction(x)
        spatial_att = torch.sigmoid(x_reduced)

        x1 = self.local_conv1(x_reduced) * spatial_att
        x2 = self.local_conv2(x1) * spatial_att
        x3 = self.local_conv3(x2) * spatial_att
        x4 = self.local_conv4(x3) * spatial_att

        x_concat = torch.concat([x1, x2, x3, x4], 1)
        x_fused = self.conv_fusion(x_concat)
        x_fused = self.se(x_fused)
        x_fused = self.ca(x_fused)
        return x_fused + x


class DPD(nn.Module):
    """Detail Preservation Downsampling."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv_fusion = Conv(c1 * 5, c2, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.concat([x0, x1, x2, x3, self.maxpool(x)], 1)
        x = self.conv_fusion(x)
        return x
    

class ICEM(nn.Module):
    """Inter-layer Cross Encoding Module."""

    def __init__(self, c_low, c_high, r=4):
        super().__init__()
        c_ = max(c_high // r, 4)
        self.dpd = DPD(c_low, c_low)

        self.conv_block = nn.Sequential(
            Conv(c_high, c_, 3, 1),
            nn.Conv2d(c_, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.modify_low = Conv(c_low, c_high, 1, 1)
        self.modify_high = Conv(c_high, c_high, 1, 1)

        self.conv_sum = Conv(c_high, c_high, 1, 1)

    def forward(self, x):
        x_low, x_high = x
        x_low = self.dpd(x_low)
        
        att_low = self.conv_block(x_high)
        x_low = self.modify_low(x_low * att_low)

        att_high = self.conv_block(x_low)
        x_high = self.modify_high(x_high * att_high)

        x = self.conv_sum(x_high + x_low)
        return x


class Flatten(nn.Module):
    """Flatten Operation."""

    def forward(self, x):
        return x.view(x.size(0), -1)
    

class IGDM(nn.Module):
    """Inter-layer Guided Decoding Module."""

    def __init__(self, c_low, c_high, r=4):
        super().__init__()
        c_ = max(c_low // r, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Sequential(
            Flatten(),
            nn.Linear(c_low, c_),
            nn.ReLU(),
            nn.Linear(c_, c_low))
        self.fc_g = nn.Sequential(
            Flatten(),
            nn.Linear(c_high, c_),
            nn.ReLU(),
            nn.Linear(c_, c_low))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_low, x_high = x
        avg_pool_x = self.avg_pool(x_low)
        channel_att_x = self.fc_x(avg_pool_x)
        avg_pool_g = self.avg_pool(x_high)
        channel_att_g = self.fc_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        channel_att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        out = self.relu(x_low * channel_att)
        return out
    