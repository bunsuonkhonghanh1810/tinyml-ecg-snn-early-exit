import torch
import torch.nn as nn
from src.models.lif_neuron import LIFCell

class SpikingResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, beta=0.85):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.lif1 = LIFCell(beta=beta)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.lif2 = LIFCell(beta=beta)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, v1=None, v2=None):
        identity = self.shortcut(x)

        c1 = self.bn1(self.conv1(x))
        s1, v1 = self.lif1(c1, v1)

        c2 = self.bn2(self.conv2(s1))
        combined_current = c2 + identity
        s2, v2 = self.lif2(combined_current, v2)

        return s2, v1, v2

class SpikingResNet1D(nn.Module):
    def __init__(self, timesteps=8, num_classes=5, beta=0.85):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes

        self.stem_conv = nn.Conv1d(1, 32, kernel_size=11, stride=1, padding=5, bias=False)
        self.stem_bn = nn.BatchNorm1d(32)
        self.stem_lif = LIFCell(beta=beta)

        self.stage1 = SpikingResBlock1D(32, 32, kernel_size=7, stride=1, beta=beta)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.stage2 = SpikingResBlock1D(32, 64, kernel_size=7, stride=1, beta=beta)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.stage3 = SpikingResBlock1D(64, 128, kernel_size=5, stride=1, beta=beta)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.stage4 = SpikingResBlock1D(128, 128, kernel_size=5, stride=1, beta=beta)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_spikes=False):
        batch_size = x.size(0)
        stem_current = self.stem_bn(self.stem_conv(x))

        v_stem = None
        v_s1_1, v_s1_2 = None, None
        v_s2_1, v_s2_2 = None, None
        v_s3_1, v_s3_2 = None, None
        v_s4_1, v_s4_2 = None, None

        features_sum = torch.zeros(batch_size, 128, device=x.device, dtype=x.dtype)

        stem_spikes_list = [] if return_spikes else None
        s1_spikes_list = [] if return_spikes else None
        s2_spikes_list = [] if return_spikes else None
        s3_spikes_list = [] if return_spikes else None
        s4_spikes_list = [] if return_spikes else None

        for _ in range(self.timesteps):
            s_stem, v_stem = self.stem_lif(stem_current, v_stem)

            s1, v_s1_1, v_s1_2 = self.stage1(s_stem, v_s1_1, v_s1_2)
            p1 = self.pool1(s1)

            s2, v_s2_1, v_s2_2 = self.stage2(p1, v_s2_1, v_s2_2)
            p2 = self.pool2(s2)

            s3, v_s3_1, v_s3_2 = self.stage3(p2, v_s3_1, v_s3_2)
            p3 = self.pool3(s3)

            s4, v_s4_1, v_s4_2 = self.stage4(p3, v_s4_1, v_s4_2)

            pooled = self.global_pool(s4).squeeze(-1)
            features_sum = features_sum + pooled

            if return_spikes:
                stem_spikes_list.append(s_stem.detach())
                s1_spikes_list.append(s1.detach())
                s2_spikes_list.append(s2.detach())
                s3_spikes_list.append(s3.detach())
                s4_spikes_list.append(s4.detach())

        features_mean = features_sum / self.timesteps
        logits = self.classifier(self.dropout(features_mean))

        if return_spikes:
            spike_record = {
                "stem": torch.stack(stem_spikes_list, dim=1),
                "s1": torch.stack(s1_spikes_list, dim=1),
                "s2": torch.stack(s2_spikes_list, dim=1),
                "s3": torch.stack(s3_spikes_list, dim=1),
                "s4": torch.stack(s4_spikes_list, dim=1),
            }
            return logits, spike_record

        return logits
