import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


# class SetBlockWrapper(nn.Module):
#     def __init__(self, forward_block):
#         super(SetBlockWrapper, self).__init__()
#         self.forward_block = forward_block

#     def forward(self, x, *args, **kwargs):
#         """
#             In  x: [n, c_in, s, h_in, w_in]
#             Out x: [n, c_out, s, h_out, w_out]
#         """
#         n, c, s, h, w = x.size()
#         x = self.forward_block(x.transpose(
#             1, 2).reshape(-1, c, h, w), *args, **kwargs)
#         output_size = x.size()
#         return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()

class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block, is_3d=False):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block
        self.is_3d = is_3d

    def forward(self, x, *args, **kwargs):
        """
        x: [B, C, T, H, W]
        - For 2D block: reshape to [B*T, C, H, W], run forward, reshape back
        - For 3D block: run directly
        """
        if self.is_3d:
#             print("特征图特征大小",x.shape)
            return self.forward_block(x, *args, **kwargs)
        else:
            B, C, T, H, W = x.shape
            x = x.transpose(1, 2).reshape(B * T, C, H, W)
            x = self.forward_block(x, *args, **kwargs)
            C_out, H_out, W_out = x.shape[1:]
            return x.reshape(B, T, C_out, H_out, W_out).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """

        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class FocalConv2d(nn.Module):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


class GaitAlign(nn.Module):
    """
        GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
        ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
        Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
    """
    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
        self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)

    def forward(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        n, c, h, w = feature_map.size()
        # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
        w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(-1)  # [n, c, h]
        _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).float().sum(-1)  # [n, c]
        h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
                 [0]).float().sum(-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = torch.clamp(p1, min=0)  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = torch.clamp(w_left, min=0., max=w+2*width_p)
        w_right = torch.clamp(w_right, min=0., max=w+2*width_p)

        boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
        # index of bbox in batch
        box_index = torch.arange(n, device=feature_map.device)
        rois = torch.cat([box_index.view(-1, 1), boxes], -1)
        crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
        return crops


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


'''
Modifed from https://github.com/BNU-IVC/FastPoseGait/blob/main/fastposegait/modeling/components/units
'''

class Graph():
    """
    # Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
    """
    def __init__(self, joint_format='coco', max_hop=2, dilation=1):
        self.joint_format = joint_format
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.joint_format == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            parts = [
                np.array([5, 7, 9]),                      # left_arm
                np.array([6, 8, 10]),                     # right_arm
                np.array([11, 13, 15]),                   # left_leg
                np.array([12, 14, 16]),                   # right_leg
                np.array([0, 1, 2, 3, 4]),                # head
            ]

        elif self.joint_format == 'coco-no-head':
            num_node = 12
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1),
                             (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
                             (6, 8), (8, 10), (7, 9), (9, 11)]
            self.edge = self_link + neighbor_link
            self.center = 0
            connect_joint = np.array([3,1,0,2,4,0,6,8,10,7,9,11])
            parts =[
                np.array([0, 2, 4]),       # left_arm
                np.array([1, 3, 5]),       # right_arm
                np.array([6, 8, 10]),      # left_leg
                np.array([7, 9, 11])       # right_leg
            ]

        elif self.joint_format =='alphapose' or self.joint_format =='openpose':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),               # left_arm
                np.array([2, 3, 4]),               # right_arm
                np.array([11, 12, 13]),            # left_leg
                np.array([8, 9, 10]),              # right_leg
                np.array([0, 1, 14, 15, 16, 17]),  # head
            ]

        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            raise ValueError('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


class TemporalBasicBlock(nn.Module):
    """
        TemporalConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False,reduction=0,get_res=False,tcn_stride=False):
        super(TemporalBasicBlock, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x


class TemporalBottleneckBlock(nn.Module):
    """
        TemporalConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4,get_res=False, tcn_stride=False):
        super(TemporalBottleneckBlock, self).__init__()
        tcn_stride =False
        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction
        if get_res:
            if tcn_stride:
                stride =2
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (2,1)),
                nn.BatchNorm2d(channels),
            )
            tcn_stride= True
        else:
            if not residual:
                self.residual = lambda x: 0
            elif stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, (2,1)),
                    nn.BatchNorm2d(channels),
                )
                tcn_stride= True

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        if tcn_stride:
            stride=2
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)
        return x



class SpatialGraphConv(nn.Module):
    """
        SpatialGraphConv_Basic_Block
        Arxiv: https://arxiv.org/abs/1801.07455
        Github: https://github.com/yysijie/st-gcn
    """
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v).contiguous()

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x

class SpatialBasicBlock(nn.Module):
    """
        SpatialGraphConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False,reduction=0):
        super(SpatialBasicBlock, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x

class SpatialBottleneckBlock(nn.Module):
    """
        SpatialGraphConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """

    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=4):
        super(SpatialBottleneckBlock, self).__init__()

        inter_channels = out_channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)

        return x

class SpatialAttention(nn.Module):
    """
    This class implements Spatial Transformer. 
    Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
    """
    def __init__(self, in_channels, out_channel, A, num_point, dk_factor=0.25, kernel_size=1, Nh=8, num=4, stride=1):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = int(dk_factor * out_channel)
        self.dv = int(out_channel)
        self.num = num
        self.Nh = Nh
        self.num_point=num_point
        self.A = A[0] + A[1] + A[2]
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
                                    stride=stride,
                                    padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
        # The multiplication can also be divided (multi_matmul) in case of space problems

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q*(dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

from einops import rearrange
class ParallelBN1d(nn.Module):
    def __init__(self, parts_num, in_channels, **kwargs):
        super(ParallelBN1d, self).__init__()
        self.parts_num = parts_num
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, p]
        '''
        x = rearrange(x, 'n c p -> n (c p)')
        x = self.bn1d(x)
        x = rearrange(x, 'n (c p) -> n c p', p=self.parts_num)
        return x
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockP3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockP3D, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu  = nn.ReLU(inplace=True)
        
        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, stride), 
                norm_layer2d(planes), 
                nn.ReLU(inplace=True)
            )
        )

        self.conv2 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(planes, planes), 
                norm_layer2d(planes), 
            )
        )

        self.shortcut3d = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.sbn        = norm_layer3d(planes)

        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.relu(out + self.sbn(self.shortcut3d(out)))
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=[1, 1, 1],  downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        assert stride[0] in [1, 2, 3]
        if stride[0] in [1, 2]: 
            tp = 1
        else:
            tp = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=[tp, 1, 1], bias=False)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1], bias=False)
        self.bn2   = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

"""
    这部分用来存放调整主干网络需要用到的模块：
    用于时间维度上的卷积DynamicFeatureBranch，
    用于时间卷积后与原图像的融合FeatureFusion
"""
def pairwise_temporal_diff(x):
    B, C, T, H, W = x.shape
    if T < 2:
        return x  # ❗ 若帧数太小就跳过差分
    if T % 2 != 0:
        x = x[:, :, :-1, :, :]  # 剪掉最后一帧
    x1 = x[:, :, 0::2, :, :]
    x2 = x[:, :, 1::2, :, :]
    return x2 - x1

class DynamicFeatureBranch(nn.Module):
    def __init__(self, conv2_block2, conv2_block3, conv2_block4):
        super().__init__()
        self.conv2 = conv2_block2  # ✅ 缺失的部分
        self.conv3 = conv2_block3
        self.conv4 = conv2_block4
    def forward(self, x):  # x 是 layer1 输出
        
        print('x分支网络起始维度大小：',x.shape)
        x = pairwise_temporal_diff(x)  # T -> T/2
#         print('x的维度:',x.shape)
        x = self.conv2(x)

        x = pairwise_temporal_diff(x)  # T/2 -> T/4
        print('经过第二次差分的维度：',x.shape)
        x = self.conv3(x)

        x = self.conv4(x)

        return x
"""

重构分支为动态渐进式分支： ProgressiveFusionBranch

"""

class ProgressiveFusionBranch(nn.Module):
    def __init__(self, conv2, conv3, conv4, fuse2, fuse3, fuse4):
        super().__init__()
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.fuse2 = fuse2
        self.fuse3 = fuse3
        self.fuse4 = fuse4

    def forward(self, x_stage1, layer2_block, layer3_block, layer4_block):
        # 分支 stage 1：差分 + 卷积
        x_dyn2 = pairwise_temporal_diff(x_stage1)
        x_dyn2 = self.conv2(x_dyn2)

        # 融合分支输出和主干 Layer2 输出，作为 Layer3 输入
        x_fused2 = self.fuse2(layer2_block, x_dyn2)

        # 分支 stage 2：差分 + 卷积
        x_dyn3 = pairwise_temporal_diff(x_fused2)
        x_dyn3 = self.conv3(x_dyn3)

        # 融合分支输出和主干 Layer3 输出，作为 Layer4 输入
        x_fused3 = self.fuse3(layer3_block, x_dyn3)

        # 分支 stage 3：不差分，直接卷积
        x_dyn4 = self.conv4(x_fused3)

        # 最终融合：与 Layer4 输出融合作为模型输出
        x_fused4 = self.fuse4(layer4_block, x_dyn4)

        return x_fused4

# class FeatureFusion(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.fuse = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.bn = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x_main, x_dyn):
#         # 对齐时间维（T），以便拼接
#         T_main = x_main.shape[2]
#         T_dyn = x_dyn.shape[2]

#         if T_main > T_dyn:
#             x_main = x_main[:, :, :T_dyn, :, :]  # 裁剪主干 T
#         elif T_main < T_dyn:
#             x_dyn = x_dyn[:, :, :T_main, :, :]  # 裁剪分支 T

#         x = torch.cat([x_main, x_dyn], dim=1)  # [B, C1+C2, T, H, W]
#         x = self.relu(self.bn(self.fuse(x)))
# #         print('特征融合后的特征大小：',x.shape)
#         return x


class FeatureFusion(nn.Module):
    def __init__(self, in1, in2, out):
        super().__init__()
        self.fuse = nn.Conv3d(in1 + in2, out, kernel_size=1)
        self.bn = nn.BatchNorm3d(out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        T1, T2 = x1.shape[2], x2.shape[2]
        if T1 > T2:
            x1 = x1[:, :, :T2]
        elif T2 > T1:
            x2 = x2[:, :, :T1]
        x = torch.cat([x1, x2], dim=1)
        return self.relu(self.bn(self.fuse(x)))

"""
更轻量的融合模块：FusionModule
"""
class FusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels * 2, channels, kernel_size=1)

    def forward(self, x1, x2):
        # 自动裁剪时间维度 T 对齐
        T1, T2 = x1.shape[2], x2.shape[2]
        if T1 > T2:
            x1 = x1[:, :, :T2]
        elif T2 > T1:
            x2 = x2[:, :, :T1]
        
        x = torch.cat([x1, x2], dim=1)
#         print("x.shape:",x.shape)
        return self.conv(x)

"""
新重构分支模块
"""
class ProgressiveFusionBranch(nn.Module):
    def __init__(self, conv2, conv3, conv4, fuse2, fuse3, fuse4, layer3, layer4):
        super().__init__()
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.fuse2 = fuse2
        self.fuse3 = fuse3
        self.fuse4 = fuse4
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, x1, x2):
        d2 = pairwise_temporal_diff(x1)
        d2_out = self.conv2(d2)
        fused2 = self.fuse2(d2_out, x2)

        x3_main = self.layer3(fused2)
        x2_diff = pairwise_temporal_diff(x2)
        d2_out_diff = pairwise_temporal_diff(d2_out)
#         print("x2_diff:",x2_diff.shape)
#         print("d2_out_diff:",d2_out_diff.shape)
        d3 = self.fuse2(pairwise_temporal_diff(x2), pairwise_temporal_diff(d2_out))
        d3_out = self.conv3(d3)
#         print("x3_main:",x3_main.shape)
#         print("d3_out:",d3_out.shape)
        x3_out = self.fuse3(x3_main, d3_out)

        x4_main = self.layer4(x3_out)
        d4 = torch.cat([pairwise_temporal_diff(x3_out), pairwise_temporal_diff(d3_out)], dim=2)
        d4_out = self.conv4(d4)
        x4_out = self.fuse4(x4_main, d4_out)

        return fused2,x3_out,x4_out

"""
多尺度融合模块
"""

class MultiScaleTemporalFusion(nn.Module):
    def __init__(self):
        super(MultiScaleTemporalFusion, self).__init__()  # 修改为正确的类名
        
        # 用1x1卷积对齐通道数为512
        self.e1_proj = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.e2_proj = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.x_dyn_proj = nn.Identity()  # x_dyn 已是512通道，无需投影

        # 下采样e1的空间分辨率从(32, 22) -> (16, 11)
        self.downsample_e1 = nn.AdaptiveAvgPool2d((16, 11))
        self.fusion_conv = nn.Conv2d(
            in_channels=512*3,   # 你拼接后的通道数
            out_channels=512,    # 输出的目标通道数，通常回到原始通道数
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, e1, e2, x_dyn):
        # squeeze temporal 维度，去掉第2维（size=1）
        e1 = e1.squeeze(2)  # [B, 128, 32, 22]
        e2 = e2.squeeze(2)  # [B, 256, 16, 11]
        x_dyn = x_dyn.mean(2)  # [B, 512, 16, 11] -> 对时间维求均值

        # 对通道维度进行对齐
        e1 = self.e1_proj(e1)  # [B, 512, 32, 22]
#         print(f"After e1 dtype: {e1.dtype}")  
        e2 = self.e2_proj(e2)  # [B, 512, 16, 11]

        # 将 e1 的空间维度对齐为(16, 11)，其余特征尺寸已经匹配
        e1 = self.downsample_e1(e1)  # [B, 512, 16, 11]
#         print(f"After downsample dtype: {e1.dtype}")  
        # e2 和 x_dyn 本身已经是 [B, 512, 16, 11]，无需修改

        # 特征融合：将三个特征在通道维度上拼接
        fused_feat = torch.cat([e1, e2, x_dyn], dim=1)  # [B, 512*3, 16, 11]
        # 使用 1x1 卷积将通道数从 1536 压缩到 512
        conv_layer = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=1)
        fused_feat = self.fusion_conv(fused_feat)  # [32, 512, 16, 11]
#         print(f"After conv_layer dtype: {fused_feat.dtype}")  
        fused_feat = fused_feat.unsqueeze(2)  # [32, 512, 1, 16, 11]
        return fused_feat

    
    
"""
辅助分支模块代码+FC映射
"""

# class SupervisedBranch(nn.Module):
#     def __init__(self, in_channels, embedding_dim=256):
#         super().__init__()
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
#         self.bn = nn.BatchNorm2d(embedding_dim)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.gap(x)  # [n, c, 1, 1]
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x.view(x.size(0), -1)  # [n, embedding_dim]


class SupervisedFCBranch(nn.Module):
    def __init__(self, in_channels, embedding_dim=256):
        super(SupervisedFCBranch, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [N, C, H, W]
        x = self.global_pool(x)  # [N, C, 1, 1]
        x = x.view(x.size(0), -1)  # [N, C]
        x = self.fc(x)             # [N, out_channels]
        x = self.bn(x)
        x = self.relu(x)
        return x


    
"""
    e1+x_dyn 和e2+x_dyn 融合的辅助监督分支。由主干动态特征引导浅层、中层特征进行学习，类似蒸馏模型的感觉。
"""
class SupervisedFusionBranch(nn.Module):
    def __init__(self, in_channels, embedding_dim=256):
        super(SupervisedFusionBranch, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.unsqueeze(-1)
    
    
"""
正金字塔融合思想
"""
class PyramidFusionBlock(nn.Module):
    def __init__(self, in_channels_e1, in_channels_e2, in_channels_xdyn, out_channels=512,
                 target_size=(16, 11)):
        super(PyramidFusionBlock, self).__init__()
        self.target_size = target_size

        # 通道对齐
        self.e1_proj = nn.Sequential(
            nn.Conv2d(in_channels_e1, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.e2_proj = nn.Sequential(
            nn.Conv2d(in_channels_e2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.xdyn_proj = nn.Identity()  # 默认 x_dyn 已是 out_channels

        # 空间尺寸对齐
        self.downsample_e1 = nn.AdaptiveAvgPool2d(target_size)

        # 融合卷积
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, e1, e2, x_dyn):
        e1 = e1.squeeze(2)  # [B, C1, H1, W1]
        e2 = e2.squeeze(2)  # [B, C2, H2, W2]
        x_dyn = x_dyn.mean(2)  # [B, C3, H2, W2]

        # 通道统一
        e1 = self.e1_proj(e1)  # [B, out, 32, 22]
        e2 = self.e2_proj(e2)  # [B, out, 16, 11]
        x_dyn = self.xdyn_proj(x_dyn)  # [B, out, 16, 11]

        # 尺寸对齐
        e1 = self.downsample_e1(e1)  # [B, out, 16, 11]

        # 正金字塔残差式融合
        e2_fused = e2 + x_dyn  # 中层增强
        x_dyn_fused = x_dyn + e2 + e1  # 高层增强

        fused = torch.cat([e1, e2_fused, x_dyn_fused], dim=1)  # [B, out*3, 16, 11]
        out = self.fusion_conv(fused)  # [B, out, 16, 11]
        return out.unsqueeze(2)  # [B, out, 1, 16, 11]
