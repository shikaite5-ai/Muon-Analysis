import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    # 这里近似笔记里的 msign(G)=UV^T：不显式做 SVD，而用多项式迭代改变奇异值、保留左右奇异方向。
    # ndim>=2 是要求 G 至少有两个维度，因为这个函数把最后两个维度当成矩阵的行、列。如 G.shape = [8, 4096, 1024]，就是有 8 个 4096x1024 的矩阵。
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    # 每个元素从 32 位浮点数转为 16 位 bfloat16，节省显存和通信带宽；Newton-Schulz 迭代在 bfloat16 下仍然稳定。
    X = G.bfloat16()
    # 让较短的那个维度成为 X 的行维，这样 X @ X.mT 产生的方阵更小。
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 用 Frobenius 范数缩放到适合迭代的数值范围（注：此处不是精确计算谱范数）。分母即 |M|_F = sqrt(sum(M_ij^2))，加上 1e-7 避免除零。
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # 五次 Newton–Schulz 多项式：A=XX^T，X <- aX+b(XX^T)X+c(XX^T)^2X。
    # 在 SVD 坐标下，这相当于对各奇异值施加同一个标量多项式，逐步压平强弱差异；
    # 有限步数时只是近似正交化，笔记中的理想 UV^T 是理解目标，不是本函数逐位保证的结果。
    for _ in range(steps):  # 这里的 A,B,C 是迭代公式的中间变量，等价于笔记里完整的矩阵多项式展开式。
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# Muon 核心更新函数：输入梯度和历史动量，输出 Muon 更新方向。
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    # 更新动量，指数滑动平均动量：momentum <- beta*momentum + (1-beta)*grad，对应笔记里的公式 m_t = beta*m_{t-1} + g_t，这里多了个 (1-beta) 归一化因子。
    momentum.lerp_(grad, 1 - beta)
    # Nesterov 前瞻方向：grad.lerp_(momentum,beta)=(1-beta)*grad+beta*momentum；
    # 关闭时则直接用动量。注意 grad.lerp_ 是原地操作，会改写传入的梯度张量。
    # 选出本次要处理的更新方向，把当前梯度和刚刚更新好的动量再插值一次：
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        # 把四维卷积 [输出通道, 输入通道, 高, 宽] 按输出通道摊平为二维矩阵，好让后面的 Newton–Schulz 方法能按矩阵处理。
        update = update.view(len(update), -1)
    # 用 Newton–Schulz 将动量/前瞻梯度矩阵近似正交化，近似矩阵符号方向。
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # 对长宽比作补偿：乘一个由矩阵形状决定的缩放系数，让不同矩阵形状的更新尺度更可比；笔记中没有这部分。
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update

# ↑ update 的数值由梯度的动量经过 Newton–Schulz 变换得到，可以近似理解为“保留主要奇异方向、让不同方向的强弱更均衡”后的矩阵


class Muon(torch.optim.Optimizer):  # 继承 PyTorch 的通用优化器基类 torch.optim.Optimizer
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)    # 优化器默认参数
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)  # 检查非空
        # 按参数形状排序，较大的形状排在前面。这是为了配合后续的分布式参数分配和通信。
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups: # 遍历优化器中的每个参数组
            params = group["params"]    # 取出当前组中的参数列表
            # 分布式版本按 rank 轮转分配矩阵：各 rank 计算自己负责参数的更新，再 all_gather 同步参数。
            # 下面这一行是让分布式通信中 每次参与的参数块数量一致，先给参数列表补上一些占位张量。这样后面的 all_gather 才能有足够的输出位置
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:  # 以进程总数为步长遍历参数列表，base_i 是当前这一批参数的起始位置
                if base_i + dist.get_rank() < len(params):  # 确保索引还在真实参数范围内，而不是落到补齐的占位区域
                    p = params[base_i + dist.get_rank()]    # 取出当前 rank 负责更新的参数 p
                    if p.grad is None:
                        # continue
                        # 该 rank 即使没有梯度也用零梯度占位，使所有 rank 仍按相同顺序参与同步。
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]   # 取得这个参数对应的优化器状态，包括历史动量等
                    if len(state) == 0: # 首次调用
                        # 每个参数单独保存其历史动量；首次调用从零开始。
                        state["momentum_buffer"] = torch.zeros_like(p)
                    # 调用 Muon 核心更新函数
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    # 解耦权重衰减先缩小参数
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    # 沿负 Muon 更新方向走学习率大小的一步
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                # 收集各 rank 更新后的参数块，确保每个 rank 上的模型副本一致。
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

# 非分布式版本的 Muon 优化器，适合单机单卡训练场景。它的核心更新逻辑与分布式版本一致，只是省去了参数分片和 all_gather 通信。
class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                # 单设备路径的核心方向算法相同，只省去分片和 all_gather 通信。
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    # 辅助 Adam 路径：一阶矩平均梯度，二阶矩平均逐元素梯度平方。
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    # 偏差修正抵消训练早期从零初始化导致的滑动平均偏小。
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    # 与 Muon 的矩阵奇异方向归一化不同，这里按坐标用 sqrt(v)+eps 缩放。
    return buf1c / (buf2c.sqrt() + eps)


# “混合优化器”：同一个优化器里，指定给 Muon 的参数用 Muon 更新，其余参数用 Adam 更新。
# 通过参数组里的 use_muon=True/False 来选择。
class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # 隐藏层矩阵权重通常放这里；Muon 将每个参数整体视作矩阵处理。
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # embedding、输出头、偏置/增益等不适合矩阵正交化的参数走辅助 Adam。
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # 混合优化器的 Muon 分支：与 Muon.step 一样分片计算、再 all_gather 同步。
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                # 非 Muon 参数走逐元素 Adam：每个坐标分别积累二阶矩和有效步长。
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # 单设备混合版本按 use_muon 分流，Muon 核心更新与分布式版本一致。
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
