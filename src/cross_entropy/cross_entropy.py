import torch


def cross_entropy(y_pred, y_true):
    """
    手动实现交叉熵损失 (Cross Entropy Loss)，一般是 logits:(N,C), label:(N,), N 代表有N个样本，C代表有C个类别

    参数:
    y_pred: torch.Tensor, 模型的原始输出 (还未经过 softmax), 形状 (batch_size, num_classes)
    y_true: torch.Tensor, 真实标签索引, 形状 (batch_size,)

    返回:
    loss: torch.Tensor, 平均损失值
    """
    # 1. softmax
    exp_y_pred = torch.exp(y_pred)
    sum_exp_y_pred = torch.sum(exp_y_pred, dim=1, keepdim=True)
    softmax_y_pred = exp_y_pred / sum_exp_y_pred

    # 2. 提取真实标签对应的概率
    batch_size = y_pred.shape[0]
    probs = softmax_y_pred[
        torch.arange(batch_size), y_true
    ]  # A=[[0,2,3],[4,5,6]]指的是取出A[0][2],A[2][3],[3,0]，这是配对索引
    # probs的形状是(batch_size,)

    # 3. 计算损失
    loss = -torch.log(probs)

    # 4. 平均损失
    return torch.mean(loss)


def cross_entropy_stable(y_pred, y_true):
    """
    稳定版交叉熵损失 (Cross Entropy Loss)，使用 log-sum-exp 技巧避免数值溢出

    公式推导：
    CE = -log(softmax(x)_k)
       = -log(exp(x_k) / sum(exp(x_j)))
       = -x_k + log(sum(exp(x_j)))

    稳定版（减去max）：
    CE = -(x_k - max) + log(sum(exp(x_j - max)))

    参数:
    y_pred: torch.Tensor, 模型的原始输出 (logits), 形状 (batch_size, num_classes)
    y_true: torch.Tensor, 真实标签索引, 形状 (batch_size,)

    返回:
    loss: torch.Tensor, 平均损失值
    """
    # 1. 减去最大值，防止 exp 溢出
    # max_y = torch.max(y_pred, dim=1, keepdim=True).values
    # y_shifted = y_pred - max_y  # 注意：这里不要 exp！保持在 log 空间

    # # 2. 计算 log(sum(exp(shifted)))
    # log_sum_exp = torch.log(torch.sum(torch.exp(y_shifted), dim=1))  # 形状 (N,)

    # # 3. 提取正确类别的 shifted logit
    # N = y_pred.shape[0]
    # correct_logit = y_shifted[torch.arange(N), y_true]  # 形状 (N,)

    # # 4. 损失 = -correct_logit + log_sum_exp
    # loss = -correct_logit + log_sum_exp

    # return torch.mean(loss)
    max_y = torch.max(y_pred, dim=1, keepdim=True).values
    y_shifted = y_pred - max_y

    exp_y_shifted = torch.exp(y_shifted)
    exp_sum = torch.sum(exp_y_shifted, dim=1, keepdim=True)
    log_exp_sum = torch.log(exp_sum)
    print(f"y_shifted:{y_shifted}")
    print(f"exp_y_shifted:{exp_y_shifted}")
    print(f"exp_sum:{exp_sum}")
    print(f"log_exp_sum:{log_exp_sum}")

    N = y_pred.shape[0]
    probs = y_shifted[torch.arange(N), y_true]

    loss = -(probs - log_exp_sum)
    print(f"probs:{probs}")
    print(f"loss:{loss}")

    return torch.mean(loss)


def ce(y_pred: torch.Tensor, y_true: torch.Tensor):
    max_value = y_pred.max(dim=1, keepdim=True).values
    y_shifted = y_pred - max_value

    log_exp_sum = torch.log(torch.exp(y_shifted).sum(dim=1, keepdim=True))

    N = y_pred.shape[0]
    probs = y_shifted[torch.arange(N), y_true]

    loss = -(probs - log_exp_sum)

    return loss.mean()


if __name__ == "__main__":
    y_pred = torch.tensor([[1000.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_true = torch.tensor([2, 1])
    print(cross_entropy(y_pred, y_true))
    print(cross_entropy_stable(y_pred, y_true))
    print(torch.nn.functional.cross_entropy(y_pred, y_true))
    print(ce(y_pred, y_true))
