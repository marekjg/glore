from torch import nn


class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x).permute(0, 2, 1)
        return self.conv1d_2(h).permute(0, 2, 1)


class GloRe(nn.Module):
    def __init__(self, in_channels, mid_channels, N):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.N = N

        self.phi = nn.Conv2d(in_channels, mid_channels, 1)
        self.theta = nn.Conv2d(in_channels, N, 1)
        self.gcn = GCN(N, mid_channels)
        self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        mid_channels = self.mid_channels
        N = self.N

        B = self.theta(x).view(batch_size, N, -1)
        x_reduced = self.phi(x).view(batch_size, mid_channels, h * w)
        x_reduced = x_reduced.permute(0, 2, 1)
        v = B.bmm(x_reduced)

        z = self.gcn(v)
        y = B.permute(0, 2, 1).bmm(z).permute(0, 2, 1)
        y = y.view(batch_size, mid_channels, h, w)
        x_res = self.phi_inv(y)

        return x + x_res
