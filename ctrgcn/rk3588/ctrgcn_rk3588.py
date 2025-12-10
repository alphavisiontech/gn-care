import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv_RK3588(nn.Module):
    """RK3588-compatible multi-scale temporal convolution with reduced channels"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilations=[1,2], residual=True, residual_kernel_size=1):
        super().__init__()
        
        # Reduce number of branches to fit RK3588 constraints
        self.num_branches = len(dilations) + 2  # dilations + max_pool + 1x1
        assert out_channels % self.num_branches == 0, '# out channels should be multiples of # branches'
        
        branch_channels = out_channels // self.num_branches
        
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, 
                           stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, 
                                       kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class CTRGC_RK3588(nn.Module):
    """RK3588-compatible Channel-wise Topology Refinement Graph Convolution"""
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC_RK3588, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Adjust channels for RK3588 constraints
        if in_channels == 3:
            self.rel_channels = 4  # Reduced from 8
            self.mid_channels = 8   # Reduced from 16
        else:
            self.rel_channels = max(1, in_channels // rel_reduction)
            self.mid_channels = max(1, in_channels // mid_reduction)
        
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_gcn_RK3588(nn.Module):
    """RK3588-compatible graph convolution unit"""
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True):
        super(unit_gcn_RK3588, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        
        # Use single CTRGC instead of multiple subsets to reduce complexity
        self.conv = CTRGC_RK3588(in_channels, out_channels)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
            
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A[0].astype(np.float32)))  # Use first subset
        else:
            self.register_buffer('A', torch.from_numpy(A[0].astype(np.float32)))
            
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
            
        y = self.conv(x, A, self.alpha)
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class TCN_GCN_unit_RK3588(nn.Module):
    """RK3588-compatible TCN-GCN unit"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, 
                 adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit_RK3588, self).__init__()
        
        self.gcn1 = unit_gcn_RK3588(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv_RK3588(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, 
            dilations=dilations, residual=False
        )
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class CTRGCN_RK3588(nn.Module):
    """Enhanced CTRGCN optimized for RK3588 with original architecture concepts"""
    def __init__(self, num_classes=2, num_joints=17, in_channels=3, base_channel=16, 
                 adaptive=True, drop_out=0.5):
        super(CTRGCN_RK3588, self).__init__()
        
        # Create adjacency matrix
        A = self._get_coco_adjacency_matrix(num_joints)
        A = np.stack([A] * 3, axis=0)  # Create 3 subsets like original
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # Network layers - keeping max channels = 32 for RK3588
        self.l1 = TCN_GCN_unit_RK3588(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit_RK3588(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit_RK3588(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit_RK3588(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)  # 32 channels max
        self.l5 = TCN_GCN_unit_RK3588(base_channel*2, base_channel*2, A, adaptive=adaptive)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channel*2, num_classes)
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        
        # Initialize
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)

    def _get_coco_adjacency_matrix(self, num_joints):
        """Create adjacency matrix for COCO skeleton"""
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (5, 11), (6, 12)  # Torso
        ]
        
        A = np.eye(num_joints)
        for i, j in connections:
            A[i, j] = 1
            A[j, i] = 1
        
        # Normalize
        D = np.sum(A, axis=1)
        D = np.diag(np.power(D, -0.5))
        A = D @ A @ D
        
        return A

    def forward(self, x):
        # Input: [N, C, T, V] where C=3, T=seq_len, V=num_joints
        N, C, T, V = x.size()
        
        # Data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # [N, C, T, V]
        
        # Forward through layers
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        # Global pooling and classification
        x = self.global_pool(x)  # [N, C, 1, 1]
        x = x.view(x.size(0), -1)  # [N, C]
        x = self.drop_out(x)
        x = self.fc(x)
        
        return x