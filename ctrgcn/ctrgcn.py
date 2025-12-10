import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def conv_init(conv):
    """Initialize convolutional layer"""
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """Initialize batch normalization layer"""
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class Graph:
    """Graph class for skeleton data representation"""
    
    def __init__(self, layout='coco', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.layout = layout
        self.strategy = strategy
        
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)
    
    def __str__(self):
        return self.A
    
    def get_edge(self):
        """Get edges for the skeleton graph"""
        if self.layout == 'coco':
            self.num_node = 17
            # COCO skeleton connections (0-indexed)
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1), (0, 2),    # nose to eyes
                (1, 3), (2, 4),    # eyes to ears
                (5, 6),            # shoulders
                (5, 7), (7, 9),    # left arm
                (6, 8), (8, 10),   # right arm
                (5, 11), (6, 12),  # shoulder to hip
                (11, 12),          # hips
                (11, 13), (13, 15), # left leg
                (12, 14), (14, 16)  # right leg
            ]
            self.edge = self_link + neighbor_link
            self.center = 0  # nose as center
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
    
    def get_adjacency(self, strategy):
        """Get adjacency matrix based on strategy"""
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        
        normalize_adjacency = normalize_digraph(adjacency)
        
        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

def get_hop_distance(num_node, edge, max_hop=1):
    """Calculate hop distance matrix"""
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    
    # Compute hop distance
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    
    return hop_dis

def normalize_digraph(A):
    """Normalize adjacency matrix"""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    
    AD = np.dot(A, Dn)
    return AD

class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution"""
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        
        # Channel-wise convolutions for topology refinement
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)  # Add softmax for attention
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A, alpha=1):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        
        # Generate attention maps
        x1 = self.conv1(x).mean(-2)  # (N, rel_channels, V)
        x2 = self.conv2(x).mean(-2)  # (N, rel_channels, V)
        x3 = self.conv3(x)  # (N, out_channels, T, V)
        
        # Compute channel-wise topology refinement
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))  # (N, rel_channels, V, V)
        x1 = self.conv4(x1) * alpha + A + A.transpose(-1, -2) # (N, out_channels, V, V)
                
        # Apply attention to features
        x1 = self.softmax(x1)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        
        return x1

class unit_tcn(nn.Module):
    """Temporal Convolutional Network unit"""
    
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), 
                             padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class unit_ctrgcn(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution unit"""
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True, attention=True):
        super(unit_ctrgcn, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.adaptive = adaptive
        self.attention = attention
        self.num_subset = A.shape[0]
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        # Residual connection
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
        
        # Learnable adjacency matrices
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            self.alpha = nn.Parameter(torch.ones(self.num_subset))
        else:
            self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
            self.alpha = nn.Parameter(torch.ones(self.num_subset))
            
        # Channel attention - ensure minimum channels
        if self.attention:
            attention_channels = max(4, out_channels//4)  # Ensure at least 4 channels
            self.chn_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, attention_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(attention_channels, out_channels, 1),
                nn.Sigmoid()
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
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
            
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha[i])
            if y is None:
                y = z
            else:
                y = y + z
        
        y = self.bn(y)
        
        # Apply channel attention
        if self.attention:
            se = self.chn_attention(y)
            y = y * se
        
        # Residual connection
        y += self.down(x)
        y = self.relu(y)
        
        return y

class TCN_GCN_unit(nn.Module):
    """Combined Temporal-Graph Convolutional unit with attention"""
    
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, 
                 adaptive=True, attention=True, kernel_size=5, dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        
        # Graph convolution with CTR
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A, 
                               adaptive=adaptive, attention=attention)
        
        # Multi-scale temporal convolution
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, 
                                           kernel_size=kernel_size, stride=stride, 
                                           dilations=dilations, residual=False)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn1(x)
        x = self.tcn1(x) + res
        return self.relu(x)


class MultiScale_TemporalConv(nn.Module):
    """Multi-scale temporal convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilations=[1, 2], residual=True, residual_kernel_size=1):
        super(MultiScale_TemporalConv, self).__init__()
        
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        
        # Temporal convolution branches
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
        
        # Additional branches
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))
        
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, 
                                       stride=stride)
    
    def forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        
        out = torch.cat(branch_outs, dim=1)
        out += self.residual(x)
        return out

class TemporalConv(nn.Module):
    """Basic temporal convolution block"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                             padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CTRGCN(nn.Module):
    """Channel-wise Topology Refinement Graph Convolutional Network for Fall Detection"""
    
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, 
                 in_channels=3, drop_out=0, adaptive=True, attention=True):
        super(CTRGCN, self).__init__()
        
        if graph is None:
            # Create default COCO graph
            graph = Graph(layout='coco', strategy='spatial', max_hop=1)
        
        A = graph.A  # (3, 17, 17) for spatial strategy

        self.num_class = num_class
        self.num_point = num_point
        
        # Data normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # Network layers with progressive channel increase
        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, 
                              adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, 
                              adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, 
                              adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, 
                              adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, 
                              adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, 
                              adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, 
                              adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, 
                              adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, 
                              adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, 
                               adaptive=adaptive, attention=attention)
        
        # Classification head
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        # Initialize data batch norm
        bn_init(self.data_bn, 1)
        
        # Dropout
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
    
    def forward(self, x):
        # Input: (N, C, T, V) or (N, C, T, V, M)
        if len(x.shape) == 3:
            # Handle (N, T, V*C) input
            N, T, _ = x.size()
            x = x.view(N, T, self.num_point, 3).permute(0, 3, 1, 2)  # (N, 3, T, 17)
        
        if len(x.shape) == 4:
            N, C, T, V = x.size()
            M = 1  # Single person
            x = x.unsqueeze(-1)  # Add person dimension
        else:
            N, C, T, V, M = x.size()
        
        # Data normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        
        # Forward through GCN-TCN blocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        # Global average pooling
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)  # Average over time and person
        
        # Dropout and classification
        x = self.drop_out(x)
        return self.fc(x)
