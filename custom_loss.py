import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLoss(nn.modules.loss._Loss):
    def __init__(self, init_weight=0, size_average=None, reduce=None, reduction='mean'):
        super(SparseLoss, self).__init__(size_average, reduce, reduction)
        self._weight = init_weight

    def forward(self, input, target, alphas):
        loss1 = F.cross_entropy(input, target)
        loss2 = -F.mse_loss(alphas, torch.tensor(0.5, requires_grad=False).expand(alphas.size()).cuda())
        return loss1 + self._weight * loss2, loss1.item(), loss2.item()

    def update_weight(self, max_weight, epoch, max_epoch):
        # epoch大　→　weight大　(線形増加)
        self._weight = max_weight * epoch / max_epoch


class SparseDeeperLoss(nn.modules.loss._Loss):
    def __init__(self, init_weight=10, bias_width=0.01, steps=4, size_average=None, reduce=None, reduction='mean'):
        super(SparseDeeperLoss, self).__init__(size_average, reduce, reduction)
        self._weight = init_weight
        self._bias_width = bias_width
        self._steps = steps
        self._vertex_mat = None

    def forward(self, input, target, alphas):
        loss1 = F.cross_entropy(input, target)
        loss2 = self._compute_reg(alphas)
        return loss1 + self._weight * loss2, loss1.item(), loss2.item()

    # 正則化項の計算
    def _compute_reg(self, alphas):
        if self._vertex_mat is None:
            self._initialize_vertex_mat(alphas.size(1))

        return -F.mse_loss(alphas, self._vertex_mat)

    def _initialize_vertex_mat(self, ops_num):
        """
        bias_width=0.1のとき　=>
        tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000],
                [0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000],
                [0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]])
        """
        # for alphas_normal
        input_node_num = 2
        mat = []
        for i in range(self._steps):  # 中間ノードの数
            for j in range(input_node_num + i):  # iノードに入力されるエッジの数
                if j >= 2:
                    mat.append([0.5 - self._bias_width * (j - 1)] * ops_num)
                else:
                    mat.append([0.5] * ops_num)
        # for alphas_reduce
        k = sum(1 for i in range(self._steps) for n in range(input_node_num + i))
        mat.extend([[0.5] * ops_num] * k)
        self._vertex_mat = torch.tensor(mat, requires_grad=False).cuda()

    def update_weight(self, max_weight, epoch, max_epoch):
        # epoch大　→　weight大　(線形増加)
        self._weight = max_weight * epoch / max_epoch
