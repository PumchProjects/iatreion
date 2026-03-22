from collections import defaultdict

import torch
import torch.nn as nn

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
INIT_L = 0.0


class GradGraft(torch.autograd.Function):
    """Implement the Gradient Grafting."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""

    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def _expand_with_not(x, m, use_not):
    if not use_not:
        return x, m
    return torch.cat((x, 1 - x), dim=1), torch.cat((m, m), dim=1)


def _expand_mask_with_not(m, use_not):
    if not use_not:
        return m
    return torch.cat((m, m), dim=1)


def _coverage_ratio(m, w):
    return (m @ w + EPSILON) / (torch.sum(w, dim=0, keepdim=True) + EPSILON)


class _LogicLayerMixin:
    def _init_missing_logic(
        self,
        *,
        missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        self.missing_aware = missing_aware
        self.coverage_tau = coverage_tau
        self.coverage_kappa = coverage_kappa
        self.last_coverage = None

    def _prepare_inputs(self, x, m):
        x, m = _expand_with_not(x, m, self.use_not)
        if self.missing_aware:
            return x, m
        return x, torch.ones_like(x)

    def _gate_outputs(self, res_tilde, res_bar, m, *, soft_mask=False):
        m = _expand_mask_with_not(m, self.use_not)
        if not self.missing_aware:
            self.last_coverage = torch.ones_like(res_bar)
            gate = torch.ones_like(res_bar)
            return res_tilde, res_bar, gate

        Wb = Binarize.apply(self.W - THRESHOLD)
        coverage_bar = _coverage_ratio(m, Wb)
        coverage_tilde = _coverage_ratio(m, self.W.clamp(0.0, 1.0))
        gate_bar = (coverage_bar >= self.coverage_tau).to(res_bar.dtype)
        gate_tilde = torch.sigmoid(
            (coverage_tilde - self.coverage_tau) / self.coverage_kappa
        )
        self.last_coverage = coverage_bar
        mask_out = gate_tilde if soft_mask else gate_bar
        return res_tilde * gate_tilde, res_bar * gate_bar, mask_out


class BinarizeLayer(nn.Module):
    """Implement the feature discretization and binarization."""

    def __init__(self, n, input_dim, use_not=False, left=None, right=None):
        super().__init__()
        self.n = n
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.use_not = use_not
        if self.use_not:
            self.disc_num *= 2
        self.output_dim = self.disc_num + self.n * self.input_dim[1] * 2
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(self.output_dim)}
        self.rule_name = None

        self.register_buffer('left', left)
        self.register_buffer('right', right)

        if self.input_dim[1] > 0:
            if self.left is not None and self.right is not None:
                cl = self.left + torch.rand(self.n, self.input_dim[1]) * (
                    self.right - self.left
                )
            else:
                cl = torch.randn(self.n, self.input_dim[1])
            self.register_buffer('cl', cl)

    def forward(self, x, m):
        if self.input_dim[1] > 0:
            x_disc, x_cont = x[:, 0 : self.input_dim[0]], x[:, self.input_dim[0] :]
            m_disc, m_cont = m[:, 0 : self.input_dim[0]], m[:, self.input_dim[0] :]
            x_cont = x_cont.unsqueeze(-1)
            if self.use_not:
                x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
                m_disc = torch.cat((m_disc, m_disc), dim=1)
            binarize_res = Binarize.apply(x_cont - self.cl.t()).reshape(x.shape[0], -1)
            m_cont = m_cont.repeat_interleave(self.n, dim=1)
            return (
                torch.cat((x_disc, binarize_res, 1.0 - binarize_res), dim=1),
                torch.cat((m_disc, m_cont, m_cont), dim=1),
            )
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
            m = torch.cat((m, m), dim=1)
        return x, m

    @torch.no_grad()
    def binarized_forward(self, x, m):
        return self.forward(x, m)

    def clip(self):
        if self.input_dim[1] > 0 and self.left is not None and self.right is not None:
            self.cl.data = torch.where(
                self.cl.data > self.right, self.right, self.cl.data
            )
            self.cl.data = torch.where(
                self.cl.data < self.left, self.left, self.cl.data
            )

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])
        if self.use_not:
            for i in range(self.input_dim[0]):
                bound_name.append('~' + feature_name[i])
        if self.input_dim[1] > 0:
            for c, op in [(self.cl, '>'), (self.cl, '<=')]:
                c = c.detach().cpu().numpy()
                for i, ci in enumerate(c.T):
                    fi_name = feature_name[self.input_dim[0] + i]
                    for j in ci:
                        if mean is not None and std is not None:
                            j = j * std[fi_name] + mean[fi_name]
                        bound_name.append(f'{fi_name} {op} {j:.6f}')
        self.rule_name = bound_name


class Product(torch.autograd.Function):
    """Tensor product function."""

    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (
            X,
            y,
        ) = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""

    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (
            X,
            y,
        ) = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (
            (-1.0 / (-1.0 + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON)
        )
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, n, input_dim):
        super().__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'
        self.rid2dim = None
        self.rule2weights = None

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, m):
        return self.fc1(x * m), torch.ones(
            x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
        )

    @torch.no_grad()
    def binarized_forward(self, x, m):
        return self.forward(x, m)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)

    def l1_norm(self):
        return torch.norm(self.fc1.weight, p=1)

    def l2_norm(self):
        return torch.sum(self.fc1.weight**2)

    def get_rule2weights(self, prev_layer, skip_connect_layer):
        prev_layer = self.conn.prev_layer
        skip_connect_layer = self.conn.skip_from_layer

        always_act_pos = prev_layer.node_activation_cnt == prev_layer.forward_tot
        merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        if skip_connect_layer is not None:
            shifted_dim2id = {
                (k + prev_layer.output_dim): (-2, v)
                for k, v in skip_connect_layer.dim2id.items()
            }
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [
                    always_act_pos,
                    (
                        skip_connect_layer.node_activation_cnt
                        == skip_connect_layer.forward_tot
                    ),
                ]
            )

        Wl, bl = list(self.fc1.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        self.rid2dim = rid2dim
        self.rule2weights = sorted(
            marked.items(),
            key=lambda x: max(x[1].values()) - min(x[1].values()),
            reverse=True,
        )


class ConjunctionLayer(_LogicLayerMixin, nn.Module):
    """The novel conjunction layer is used to learn the conjunction of nodes with less time and GPU memory usage."""

    def __init__(
        self,
        n,
        input_dim,
        use_not=False,
        alpha=0.999,
        beta=8,
        gamma=1,
        *,
        missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(
            INIT_L + (0.5 - INIT_L) * torch.rand(self.input_dim, self.n)
        )

        self.node_activation_cnt = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._init_missing_logic(
            missing_aware=missing_aware,
            coverage_tau=coverage_tau,
            coverage_kappa=coverage_kappa,
        )

    def forward(self, x, m):
        res_tilde = self.continuous_forward(x, m)
        res_bar = self.binarized_forward(x, m)
        res_tilde, res_bar, m_out = self._gate_outputs(
            res_tilde, res_bar, m, soft_mask=True
        )
        return GradGraft.apply(res_bar, res_tilde), m_out

    def continuous_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        x = (1.0 - x) * m
        xl = 1.0 - 1.0 / (1.0 - (x * self.alpha) ** self.beta)
        wl = 1.0 - 1.0 / (1.0 - (self.W * self.alpha) ** self.beta)
        return 1.0 / (1.0 + xl @ wl) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        Wb = Binarize.apply(self.W - THRESHOLD)
        res = ((1 - x) * m) @ Wb
        return torch.where(res > 0, torch.zeros_like(res), torch.ones_like(res))

    def clip(self):
        self.W.data.clamp_(INIT_L, 1.0)


class DisjunctionLayer(_LogicLayerMixin, nn.Module):
    """The novel disjunction layer is used to learn the disjunction of nodes with less time and GPU memory usage."""

    def __init__(
        self,
        n,
        input_dim,
        use_not=False,
        alpha=0.999,
        beta=8,
        gamma=1,
        *,
        missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'

        self.W = nn.Parameter(
            INIT_L + (0.5 - INIT_L) * torch.rand(self.input_dim, self.n)
        )

        self.node_activation_cnt = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._init_missing_logic(
            missing_aware=missing_aware,
            coverage_tau=coverage_tau,
            coverage_kappa=coverage_kappa,
        )

    def forward(self, x, m):
        res_tilde = self.continuous_forward(x, m)
        res_bar = self.binarized_forward(x, m)
        res_tilde, res_bar, m_out = self._gate_outputs(
            res_tilde, res_bar, m, soft_mask=True
        )
        return GradGraft.apply(res_bar, res_tilde), m_out

    def continuous_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        x = x * m
        xl = 1.0 - 1.0 / (1.0 - (x * self.alpha) ** self.beta)
        wl = 1.0 - 1.0 / (1.0 - (self.W * self.alpha) ** self.beta)
        return 1.0 - 1.0 / (1.0 + xl @ wl) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        Wb = Binarize.apply(self.W - THRESHOLD)
        res = (x * m) @ Wb
        return torch.where(res > 0, torch.ones_like(res), torch.zeros_like(res))

    def clip(self):
        self.W.data.clamp_(INIT_L, 1.0)


class OriginalConjunctionLayer(_LogicLayerMixin, nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(
        self,
        n,
        input_dim,
        use_not=False,
        estimated_grad=False,
        *,
        missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.n))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None
        self._init_missing_logic(
            missing_aware=missing_aware,
            coverage_tau=coverage_tau,
            coverage_kappa=coverage_kappa,
        )

    def forward(self, x, m):
        res_tilde = self.continuous_forward(x, m)
        res_bar = self.binarized_forward(x, m)
        res_tilde, res_bar, m_out = self._gate_outputs(
            res_tilde, res_bar, m, soft_mask=True
        )
        return GradGraft.apply(res_bar, res_tilde), m_out

    def continuous_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        return self.Product.apply(1 - ((1 - x) * m).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return torch.prod(1 - ((1 - x) * m).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class OriginalDisjunctionLayer(_LogicLayerMixin, nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(
        self,
        n,
        input_dim,
        use_not=False,
        estimated_grad=False,
        *,
        missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.n))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None
        self._init_missing_logic(
            missing_aware=missing_aware,
            coverage_tau=coverage_tau,
            coverage_kappa=coverage_kappa,
        )

    def forward(self, x, m):
        res_tilde = self.continuous_forward(x, m)
        res_bar = self.binarized_forward(x, m)
        res_tilde, res_bar, m_out = self._gate_outputs(
            res_tilde, res_bar, m, soft_mask=True
        )
        return GradGraft.apply(res_bar, res_tilde), m_out

    def continuous_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        return 1 - self.Product.apply(1 - (x * m).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x, m):
        x, m = self._prepare_inputs(x, m)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - (x * m).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


def extract_rules(prev_layer, skip_connect_layer, layer, pos_shift=0):
    # dim2id = {dimension: rule_id} :
    dim2id = defaultdict(lambda: -1)
    rules = {}
    tmp = 0
    rule_list = []

    # Wb.shape = (n, input_dim)
    Wb = (layer.W.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # merged_dim2id is the dim2id of the input (the prev_layer and skip_connect_layer)
    merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
    if skip_connect_layer is not None:
        shifted_dim2id = {
            (k + prev_layer.output_dim): (-2, v)
            for k, v in skip_connect_layer.dim2id.items()
        }
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})

    for ri, row in enumerate(Wb):
        # delete dead nodes
        if (
            layer.node_activation_cnt[ri + pos_shift] == 0
            or layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot
        ):
            dim2id[ri + pos_shift] = -1
            continue
        rule = {}
        # rule[i] = (k, rule_id):
        #     k == -1: connects to a rule in prev_layer,
        #     k ==  1: connects to a rule in prev_layer (NOT),
        #     k == -2: connects to a rule in skip_connect_layer,
        #     k ==  2: connects to a rule in skip_connect_layer (NOT).
        bound = {}
        if prev_layer.layer_type == 'binarization' and prev_layer.input_dim[1] > 0:
            c = (
                torch.cat(
                    (prev_layer.cl.t().reshape(-1), prev_layer.cl.t().reshape(-1))
                )
                .detach()
                .cpu()
                .numpy()
            )
        for i, w in enumerate(row):
            # deal with "use NOT", use_not_mul = -1 if it used NOT in that input dimension
            use_not_mul = 1
            if layer.use_not:
                if i >= layer.input_dim // 2:
                    use_not_mul = -1
                i = i % (layer.input_dim // 2)

            if w > 0 and merged_dim2id[i][1] != -1:
                if prev_layer.layer_type == 'binarization' and i >= prev_layer.disc_num:
                    ci = i - prev_layer.disc_num
                    bi = ci // prev_layer.n
                    if bi not in bound:
                        bound[bi] = [i, c[ci]]
                        rule[(-1, i)] = 1  # since dim2id[i] == i in the BinarizeLayer
                    else:  # merge the bounds for one feature
                        if (
                            ci < c.shape[0] // 2 and layer.layer_type == 'conjunction'
                        ) or (
                            ci >= c.shape[0] // 2 and layer.layer_type == 'disjunction'
                        ):
                            func = max
                        else:
                            func = min
                        bound[bi][1] = func(bound[bi][1], c[ci])
                        if bound[bi][1] == c[ci]:  # replace the last bound
                            del rule[(-1, bound[bi][0])]
                            rule[(-1, i)] = 1
                            bound[bi][0] = i
                else:
                    rid = merged_dim2id[i]
                    rule[(rid[0] * use_not_mul, rid[1])] = 1

        # give each unique rule an id, and save this id in dim2id
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list


class UnionLayer(nn.Module):
    """The union layer is used to learn the rule-based representation."""

    def __init__(
        self,
        n,
        input_dim,
        use_not=False,
        use_nlaf=False,
        estimated_grad=False,
        alpha=0.999,
        beta=8,
        gamma=1,
        missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim
        self.output_dim = self.n * 2
        self.layer_type = 'union'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None
        self.node_coverage_sum = None
        self.last_coverage = None
        self.coverage_tau = coverage_tau

        if use_nlaf:  # use novel logical activation functions
            self.con_layer = ConjunctionLayer(
                self.n,
                self.input_dim,
                use_not=use_not,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                missing_aware=missing_aware,
                coverage_tau=coverage_tau,
                coverage_kappa=coverage_kappa,
            )
            self.dis_layer = DisjunctionLayer(
                self.n,
                self.input_dim,
                use_not=use_not,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                missing_aware=missing_aware,
                coverage_tau=coverage_tau,
                coverage_kappa=coverage_kappa,
            )
        else:  # use original logical activation functions
            self.con_layer = OriginalConjunctionLayer(
                self.n,
                self.input_dim,
                use_not=use_not,
                estimated_grad=estimated_grad,
                missing_aware=missing_aware,
                coverage_tau=coverage_tau,
                coverage_kappa=coverage_kappa,
            )
            self.dis_layer = OriginalDisjunctionLayer(
                self.n,
                self.input_dim,
                use_not=use_not,
                estimated_grad=estimated_grad,
                missing_aware=missing_aware,
                coverage_tau=coverage_tau,
                coverage_kappa=coverage_kappa,
            )

    def forward(self, x, m):
        con_x, con_m = self.con_layer(x, m)
        dis_x, dis_m = self.dis_layer(x, m)
        self.last_coverage = torch.cat(
            [self.con_layer.last_coverage, self.dis_layer.last_coverage], dim=1
        )
        return torch.cat([con_x, dis_x], dim=1), torch.cat([con_m, dis_m], dim=1)

    def binarized_forward(self, x, m):
        con_x = self.con_layer.binarized_forward(x, m)
        dis_x = self.dis_layer.binarized_forward(x, m)
        _, con_x, con_m = self.con_layer._gate_outputs(con_x, con_x, m)
        _, dis_x, dis_m = self.dis_layer._gate_outputs(dis_x, dis_x, m)
        self.last_coverage = torch.cat(
            [self.con_layer.last_coverage, self.dis_layer.last_coverage], dim=1
        )
        return torch.cat([con_x, dis_x], dim=1), torch.cat([con_m, dis_m], dim=1)

    def edge_count(self):
        con_Wb = Binarize.apply(self.con_layer.W - THRESHOLD)
        dis_Wb = Binarize.apply(self.dis_layer.W - THRESHOLD)
        return torch.sum(con_Wb) + torch.sum(dis_Wb)

    def l1_norm(self):
        return torch.sum(self.con_layer.W) + torch.sum(self.dis_layer.W)

    def l2_norm(self):
        return torch.sum(self.con_layer.W**2) + torch.sum(self.dis_layer.W**2)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = (
            self.node_activation_cnt
        )

        # get dim2id and rule lists of the conjunction layer and the disjunction layer
        # dim2id: dimension --> (k, rule id)
        con_dim2id, con_rule_list = extract_rules(
            prev_layer, skip_connect_layer, self.con_layer
        )
        dis_dim2id, dis_rule_list = extract_rules(
            prev_layer, skip_connect_layer, self.dis_layer, self.con_layer.W.shape[1]
        )

        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})

        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

    def get_rule_description(self, input_rule_name, wrap=False):
        """
        input_rule_name: (skip_connect_rule_name, prev_rule_name)
        """
        self.rule_name = []
        for rl, op in zip(self.rule_list, ('&', '|'), strict=True):
            for rule in rl:
                name = ''
                for i, ri in enumerate(rule):
                    op_str = f' {op} ' if i != 0 else ''
                    layer_shift = ri[0]
                    not_str = ''
                    if ri[0] > 0:  # ri[0] == 1 or ri[0] == 2
                        layer_shift *= -1
                        not_str = '~'
                    var_str = ('({})' if (wrap or not_str == '~') else '{}').format(
                        input_rule_name[2 + layer_shift][ri[1]]
                    )
                    name += op_str + not_str + var_str
                self.rule_name.append(name)
