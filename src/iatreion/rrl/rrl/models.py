import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

from iatreion.utils import logger

from .components import BinarizeLayer, LRLayer, UnionLayer


class Net(nn.Module):
    def __init__(
        self,
        dim_list,
        use_not=False,
        left=None,
        right=None,
        use_nlaf=False,
        estimated_grad=False,
        use_skip=True,
        alpha=0.999,
        beta=8,
        gamma=1,
        temperature=0.01,
        use_missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])
        self.use_skip = use_skip
        self.use_missing_aware = use_missing_aware
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim

            skip_from_layer = None
            if self.use_skip and i >= 4:
                skip_from_layer = self.layer_list[-2]
                num += skip_from_layer.output_dim

            if i == 1:
                layer = BinarizeLayer(
                    dim_list[i], num, self.use_not, self.left, self.right
                )
                layer_name = f'binary{i}'
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = f'lr{i}'
            else:
                # The first logical layer does not use NOT if the binarization layer has already used NOT
                layer_use_not = i != 2
                layer = UnionLayer(
                    dim_list[i],
                    num,
                    use_nlaf=use_nlaf,
                    estimated_grad=estimated_grad,
                    use_not=layer_use_not,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    missing_aware=use_missing_aware,
                    coverage_tau=coverage_tau,
                    coverage_kappa=coverage_kappa,
                )
                layer_name = f'union{i}'

            layer.conn = lambda: None  # create an empty class to save the connections
            layer.conn.prev_layer = (
                self.layer_list[-1] if len(self.layer_list) > 0 else None
            )
            layer.conn.is_skip_to_layer = False
            layer.conn.skip_from_layer = skip_from_layer
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True

            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x, m):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                m = torch.cat((m, layer.conn.skip_from_layer.m_res), dim=1)
                del layer.conn.skip_from_layer.x_res
                del layer.conn.skip_from_layer.m_res
            x, m = layer(x, m)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
                layer.m_res = m
        return x

    def bi_forward(self, x, m, count=False):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                m = torch.cat((m, layer.conn.skip_from_layer.m_res), dim=1)
                del layer.conn.skip_from_layer.x_res
                del layer.conn.skip_from_layer.m_res
            x, m = layer.binarized_forward(x, m)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
                layer.m_res = m
            if count and layer.layer_type != 'linear':
                layer.node_activation_cnt += torch.sum(x * m, dim=0, dtype=torch.double)
                if (
                    getattr(layer, 'node_coverage_sum', None) is not None
                    and getattr(layer, 'last_coverage', None) is not None
                ):
                    layer.node_coverage_sum += torch.sum(
                        layer.last_coverage, dim=0, dtype=torch.double
                    )
                layer.forward_tot += x.shape[0]
        return x


class RRL:
    def __init__(
        self,
        dim_list,
        use_not=False,
        writer=None,
        left=None,
        right=None,
        estimated_grad=False,
        save_model_callback=None,
        use_skip=False,
        use_nlaf=False,
        alpha=0.999,
        beta=8,
        gamma=1,
        temperature=0.01,
        use_missing_aware=False,
        coverage_tau=0.5,
        coverage_kappa=0.1,
    ):
        super().__init__()
        self.dim_list = dim_list
        self.use_not = use_not
        self.use_skip = use_skip
        self.use_nlaf = use_nlaf
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_missing_aware = use_missing_aware
        self.best_f1 = -1.0
        self.best_loss = 1e20

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.estimated_grad = estimated_grad
        self.save_model_callback = save_model_callback

        self.writer = writer

        self.net = Net(
            dim_list,
            use_not=use_not,
            left=left,
            right=right,
            use_nlaf=use_nlaf,
            estimated_grad=estimated_grad,
            use_skip=use_skip,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            use_missing_aware=use_missing_aware,
            coverage_tau=coverage_tau,
            coverage_kappa=coverage_kappa,
        )
        self.net.to(self.device)

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[:-1]:
            layer.clip()

    def edge_penalty(self):
        edge_penalty = 0.0
        for layer in self.net.layer_list[1:-1]:
            edge_penalty += layer.edge_count()
        return edge_penalty

    def l1_penalty(self):
        l1_penalty = 0.0
        for layer in self.net.layer_list[1:]:
            l1_penalty += layer.l1_norm()
        return l1_penalty

    def l2_penalty(self):
        l2_penalty = 0.0
        for layer in self.net.layer_list[1:]:
            l2_penalty += layer.l2_norm()
        return l2_penalty

    def mixed_penalty(self):
        penalty = 0.0
        for layer in self.net.layer_list[1:-1]:
            penalty += layer.l2_norm()
        penalty += self.net.layer_list[-1].l1_norm()
        return penalty

    @staticmethod
    def exp_lr_scheduler(
        optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7
    ):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(
        self,
        epoch_advance=None,
        data_loader=None,
        valid_loader=None,
        epoch=50,
        lr=0.01,
        lr_decay_epoch=100,
        lr_decay_rate=0.75,
        weight_decay=0.0,
        class_weights=None,
        log_iter=50,
        save_interval=100,
        early_stop_patience=None,
        early_stop_min_delta=0.0,
        label_smoothing=0.0,
        max_grad_norm=5.0,
    ):

        if data_loader is None:
            raise Exception('Data loader is unavailable!')

        accuracy_b = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0)

        cnt = -1
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        best_loss = 1e20  # NOTE: Please distinguish this from self.best_loss
        no_improve_checks = 0
        use_early_stop = (
            valid_loader is not None
            and early_stop_patience is not None
            and early_stop_patience > 0
        )
        early_stopped = False
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(
                optimizer,
                epo,
                init_lr=lr,
                lr_decay_rate=lr_decay_rate,
                lr_decay_epoch=lr_decay_epoch,
            )

            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 0
            for X, M, y in data_loader:
                ba_cnt += 1
                X = X.to(self.device, non_blocking=True)
                M = M.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                optimizer.zero_grad()  # Zero the gradient buffers.

                # trainable softmax temperature
                y_bar = self.net.forward(X, M) / torch.exp(self.net.t)

                loss_rrl = criterion(y_bar, y) + weight_decay * self.l2_penalty()

                ba_loss_rrl = loss_rrl.item()
                epoch_loss_rrl += ba_loss_rrl
                avg_batch_loss_rrl += ba_loss_rrl

                loss_rrl.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

                cnt += 1
                with torch.no_grad():
                    if cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                        self.writer.add_scalar(
                            'Avg_Batch_Loss_GradGrafting',
                            avg_batch_loss_rrl / log_iter,
                            cnt,
                        )
                        edge_p = self.edge_penalty().item()
                        self.writer.add_scalar('Edge_penalty/Log', np.log(edge_p), cnt)
                        self.writer.add_scalar('Edge_penalty/Origin', edge_p, cnt)
                        avg_batch_loss_rrl = 0.0

                optimizer.step()

                for param in self.net.parameters():
                    abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                    abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (
                        param.grad.numel()
                    )
                self.clip()

                if valid_loader is not None and cnt > 0 and cnt % save_interval == 0:
                    _, acc_b, f1_b = self.test(
                        test_loader=valid_loader, set_name='Validation'
                    )
                    avg_epoch_loss_rrl = epoch_loss_rrl / max(ba_cnt, 1)
                    improved = (f1_b - self.best_f1) > early_stop_min_delta
                    same_f1_better_loss = (
                        np.abs(f1_b - self.best_f1) <= early_stop_min_delta
                        and self.best_loss > avg_epoch_loss_rrl
                    )
                    if improved or same_f1_better_loss:
                        logger.info(
                            f'[bold yellow]New best model found! {self.best_f1:.2%} -> {f1_b:.2%}',
                            extra={'markup': True},
                        )
                        self.best_f1 = f1_b
                        self.best_loss = avg_epoch_loss_rrl
                        self.save_model(1.0 - acc_b, f1_b)
                        no_improve_checks = 0
                    elif use_early_stop:
                        no_improve_checks += 1
                        if no_improve_checks >= early_stop_patience:
                            logger.info(
                                f'[bold yellow]Early stopping triggered at epoch {epo}, step {cnt}: '
                                f'no validation F1 improvement > {early_stop_min_delta} '
                                f'for {early_stop_patience} checks.',
                                extra={'markup': True},
                            )
                            early_stopped = True

                    accuracy_b.append(acc_b)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar(
                            'Accuracy_RRL', acc_b, cnt // save_interval
                        )
                        self.writer.add_scalar(
                            'F1_Score_RRL', f1_b, cnt // save_interval
                        )
                    if early_stopped:
                        break

            logger.info(f'epoch: {epo}, loss_rrl: {epoch_loss_rrl}')
            if (
                valid_loader is None and epo % save_interval == 0
            ):  # use the data_loader as the valid loader
                _, acc_b, f1_b = self.test(test_loader=data_loader, set_name='Training')
                avg_epoch_loss_rrl = epoch_loss_rrl / max(ba_cnt, 1)
                if avg_epoch_loss_rrl < best_loss:
                    logger.info(
                        '[bold green]New best model found!', extra={'markup': True}
                    )
                    best_loss = avg_epoch_loss_rrl
                    self.save_model(1.0 - acc_b, f1_b)

                accuracy_b.append(acc_b)
                f1_score_b.append(f1_b)
                if self.writer is not None:
                    self.writer.add_scalar('Accuracy_RRL', acc_b, epo // save_interval)
                    self.writer.add_scalar('F1_Score_RRL', f1_b, epo // save_interval)

            if self.writer is not None:
                self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                self.writer.add_scalar(
                    'Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo
                )

            epoch_advance()
            if early_stopped:
                break

        if valid_loader is not None and self.best_f1 < 0:
            _, acc_b, f1_b = self.test(test_loader=valid_loader, set_name='Validation')
            self.best_f1 = f1_b
            self.best_loss = epoch_loss_rrl / max(ba_cnt, 1)
            self.save_model(1.0 - acc_b, f1_b)

        return epoch_histc

    @torch.no_grad()
    def predict_proba(self, test_loader):
        y_pred_b_list = []
        for X, M in test_loader:
            X = X.to(self.device, non_blocking=True)
            M = M.to(self.device, non_blocking=True)
            output = self.net.forward(X, M) / torch.exp(self.net.t)
            y_pred_b_list.append(output)
        y_pred_b = torch.cat(y_pred_b_list).softmax(dim=1).numpy(force=True)
        return y_pred_b

    @torch.no_grad()
    def test(self, test_loader=None, set_name='Validation'):
        if test_loader is None:
            raise Exception('Data loader is unavailable!')

        y_list = []
        for _X, _M, y in test_loader:
            y_list.append(y)
        y_true = torch.cat(y_list, dim=0)
        y_true = y_true.cpu().numpy().astype(int)
        data_num = y_true.shape[0]

        slice_step = data_num // 40 if data_num >= 40 else 1
        logger.debug(f'y_true: {y_true.shape} {y_true[::slice_step]}')

        y_pred_b_list = []
        for X, M, _y in test_loader:
            X = X.to(self.device, non_blocking=True)
            M = M.to(self.device, non_blocking=True)
            output = self.net.forward(X, M) / torch.exp(self.net.t)
            y_pred_b_list.append(output)

        y_pred_b = torch.cat(y_pred_b_list).softmax(dim=1).cpu().numpy()
        y_pred_b_arg = np.argmax(y_pred_b, axis=1)
        logger.debug(f'y_rrl_: {y_pred_b_arg.shape} {y_pred_b_arg[::slice_step]}')
        logger.debug(f'y_rrl: {y_pred_b.shape} {y_pred_b[::(slice_step)]}')

        accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)
        f1_score_b = metrics.f1_score(
            y_true, y_pred_b_arg, average='macro', zero_division=0
        )

        labels = list(range(self.dim_list[-1]))
        logger.debug('-' * 60)
        logger.debug(
            f'On {set_name} Set:\n\tAccuracy of RRL  Model: {accuracy_b}'
            f'\n\tF1 Score of RRL  Model: {f1_score_b}'
        )
        logger.debug(
            f'On {set_name} Set:\nPerformance of  RRL Model: \n{metrics.confusion_matrix(y_true, y_pred_b_arg, labels=labels)}\n{metrics.classification_report(y_true, y_pred_b_arg, zero_division=0)}'
        )
        logger.debug('-' * 60)

        return y_pred_b, accuracy_b, f1_score_b

    def save_model(self, *metrics):
        state_dict = {
            key: value.clone() for key, value in self.net.state_dict().items()
        }
        self.save_model_callback(self, state_dict, metrics)

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(
                    layer.output_dim, dtype=torch.double, device=self.device
                )
                if hasattr(layer, 'node_coverage_sum'):
                    layer.node_coverage_sum = torch.zeros(
                        layer.output_dim, dtype=torch.double, device=self.device
                    )
                layer.forward_tot = 0

            for x, m, _y in data_loader:
                x_bar = x.to(self.device)
                m_bar = m.to(self.device)
                self.net.bi_forward(x_bar, m_bar, count=True)

    def rule_print(
        self,
        feature_name,
        label_name,
        train_loader,
        file=sys.stdout,
        mean=None,
        std=None,
        display=True,
        metrics=None,
    ):
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception('Need train_loader for the dead nodes detection.')

        # detect dead nodes first
        if self.net.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        # for Binarize Layer
        self.net.layer_list[0].get_bound_name(
            feature_name, mean, std
        )  # layer_list[0].rule_name == bound_name

        # for Union Layer
        for i in range(1, len(self.net.layer_list) - 1):
            layer = self.net.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = (
                None
                if layer.conn.skip_from_layer is None
                else layer.conn.skip_from_layer.rule_name
            )
            wrap_prev_rule = i != 1  # do not warp the bound_name
            layer.get_rule_description(
                (skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule
            )

        # for LR Layr
        layer = self.net.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)

        if not display:
            return layer.rule2weights

        _, acc, f1 = self.test(test_loader=train_loader, set_name='Training')
        temp = torch.exp(self.net.t).item()
        print(
            'RID(et={:.4f},ft={:.4f},ev={:.4f},fv={:.4f},t={:.5f})'.format(
                1.0 - acc, f1, *metrics, temp
            ),
            end='\t',
            file=file,
        )
        for i, ln in enumerate(label_name):
            print(f'{ln}(b={layer.bl[i] / temp:.4f})', end='\t', file=file)
        if self.use_missing_aware:
            print('Support\tMeanCoverage\tTau\tRule', file=file)
        else:
            print('Support\tRule', file=file)
        for rid, w in layer.rule2weights:
            print(rid, end='\t', file=file)
            for li in range(len(label_name)):
                print(f'{w[li] / temp:.4f}', end='\t', file=file)
            now_layer = self.net.layer_list[-1 + rid[0]]
            support = (
                now_layer.node_activation_cnt[layer.rid2dim[rid]]
                / now_layer.forward_tot
            ).item()
            print(f'{support:.4f}', end='\t', file=file)
            if self.use_missing_aware:
                coverage = (
                    now_layer.node_coverage_sum[layer.rid2dim[rid]]
                    / now_layer.forward_tot
                ).item()
                print(
                    f'{coverage:.4f}\t{now_layer.coverage_tau:.4f}\t',
                    end='',
                    file=file,
                )
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        return layer.rule2weights
