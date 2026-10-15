import torch
import torch.nn as nn
from src.models.lif_neuron import LIFCell

class BaselineSNN(nn.Module):
    def __init__(self, timesteps=8, num_classes=5, beta=0.85):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.lif1 = LIFCell(beta=beta)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.lif2 = LIFCell(beta=beta)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.lif3 = LIFCell(beta=beta)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.lif4 = LIFCell(beta=beta)
        self.pool4 = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, return_spikes=False):
        batch_size = x.size(0)
        current1 = self.bn1(self.conv1(x))

        v1 = None
        v2 = None
        v3 = None
        v4 = None

        features_sum = torch.zeros(batch_size, 64, device=x.device, dtype=x.dtype)

        spikes1_list = [] if return_spikes else None
        spikes2_list = [] if return_spikes else None
        spikes3_list = [] if return_spikes else None
        spikes4_list = [] if return_spikes else None

        for _ in range(self.timesteps):
            s1, v1 = self.lif1(current1, v1)
            p1 = self.pool1(s1)

            c2 = self.bn2(self.conv2(p1))
            s2, v2 = self.lif2(c2, v2)
            p2 = self.pool2(s2)

            c3 = self.bn3(self.conv3(p2))
            s3, v3 = self.lif3(c3, v3)
            p3 = self.pool3(s3)

            c4 = self.bn4(self.conv4(p3))
            s4, v4 = self.lif4(c4, v4)
            p4 = self.pool4(s4)

            flat = p4.squeeze(-1)
            features_sum = features_sum + flat

            if return_spikes:
                spikes1_list.append(s1.detach())
                spikes2_list.append(s2.detach())
                spikes3_list.append(s3.detach())
                spikes4_list.append(s4.detach())

        features_mean = features_sum / self.timesteps
        logits = self.classifier(self.dropout(features_mean))

        if return_spikes:
            spike_record = {
                "b1": torch.stack(spikes1_list, dim=1),
                "b2": torch.stack(spikes2_list, dim=1),
                "b3": torch.stack(spikes3_list, dim=1),
                "b4": torch.stack(spikes4_list, dim=1),
            }
            return logits, spike_record

        return logits
