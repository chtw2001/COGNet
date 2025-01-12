import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 1: (voc_size, voc_size) * (voc_size, 64)
        # 2, 3: (voc_size, 64) * (64, 64)
        support = torch.mm(input, self.weight)
        # 1: (voc_size, voc_size) * (voc_size, 64)
        # 2, 3: (voc_size, 64)
        output = torch.mm(adj, support)
        if self.bias is not None:
            # 1, 2, 3: (voc_size, 64)
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        # seq_masks -> (batch_size * 환자의 최대 시퀀스, 환자의 최대 ICD9_CODE 개수). 실제 인덱스는 0, 관련 없는것은 -1e9
        
        # seqs -> (batch_size * max_visit_num, max_diag_num, 64)
        # -> (batch_size * max_visit_num, max_diag_num, 32)
        # -> (batch_size * max_visit_num, max_diag_num, 1) 
        # -> (batch_size * max_visit_num, max_diag_num)
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates + seq_masks
        # 관련 있는 약물에 대해서 확률 높게 생성
        p_attn = F.softmax(gates, dim=-1)
        # (batch_size * max_visit_num, max_diag_num, 1)
        p_attn = p_attn.unsqueeze(-1)
        # attention 가중치 곱
        h = seqs * p_attn
        # (batch_size * max_visit_num, 64)
        output = torch.sum(h, dim=1)
        return output
