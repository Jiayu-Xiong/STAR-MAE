# optim_factory.py

import json
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch import optim as optim
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

try:
    from apex.apex.optimizers import FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model,
                         weight_decay=1e-5,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.endswith(
                ".scale") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


class FP16Adam(Optimizer):
    """Implements Adam algorithm with FP16 first moment to save memory.

    This optimizer uses half-precision (FP16) for the first moment estimates (mean),
    which can save memory during training large models like ViT-G.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float]): coefficients used for computing running averages
            of gradient and its square (default: (0.9, 0.95))
        eps (float): term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay (float): weight decay coefficient (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(FP16Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()  # Ensure gradient is in FP32
                if grad.is_sparse:
                    raise RuntimeError('FP16Adam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Use FP16 for exp_avg to save memory
                    state['exp_avg'] = torch.zeros_like(p_data_fp32).half()
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(1 - beta1, grad.half())
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']

                # Update parameters
                p_data_fp32.addcdiv_(-step_size, exp_avg.float(), denom)
                p.data.copy_(p_data_fp32)

        return loss


def create_optimizer(args,
                     model,
                     get_num_layer=None,
                     get_layer_scale=None,
                     filter_bias_and_bn=True,
                     skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip,
                                          get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(
        ), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'fp16adam':
        optimizer = FP16Adam(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    else:
        if opt_lower == 'sgd' or opt_lower == 'nesterov':
            opt_args.pop('eps', None)
            optimizer = optim.SGD(
                parameters, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'momentum':
            opt_args.pop('eps', None)
            optimizer = optim.SGD(
                parameters, momentum=args.momentum, nesterov=False, **opt_args)
        elif opt_lower == 'adam':
            optimizer = optim.Adam(parameters, **opt_args)
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW(parameters, **opt_args)
        elif opt_lower == 'nadam':
            optimizer = Nadam(parameters, **opt_args)
        elif opt_lower == 'radam':
            optimizer = RAdam(parameters, **opt_args)
        elif opt_lower == 'adamp':
            optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
        elif opt_lower == 'sgdp':
            optimizer = SGDP(
                parameters, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'adadelta':
            optimizer = optim.Adadelta(parameters, **opt_args)
        elif opt_lower == 'adahessian':
            optimizer = Adahessian(parameters, **opt_args)
        elif opt_lower == 'rmsprop':
            optimizer = optim.RMSprop(
                parameters, alpha=0.9, momentum=args.momentum, **opt_args)
        elif opt_lower == 'rmsproptf':
            optimizer = RMSpropTF(
                parameters, alpha=0.9, momentum=args.momentum, **opt_args)
        elif opt_lower == 'novograd':
            optimizer = NovoGrad(parameters, **opt_args)
        elif opt_lower == 'nvnovograd':
            optimizer = NvNovoGrad(parameters, **opt_args)
        elif opt_lower == 'fusedsgd':
            opt_args.pop('eps', None)
            optimizer = FusedSGD(
                parameters, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'fusedmomentum':
            opt_args.pop('eps', None)
            optimizer = FusedSGD(
                parameters, momentum=args.momentum, nesterov=False, **opt_args)
        elif opt_lower == 'fusedadam':
            optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
        elif opt_lower == 'fusedadamw':
            optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
        elif opt_lower == 'fusedlamb':
            optimizer = FusedLAMB(parameters, **opt_args)
        elif opt_lower == 'fusednovograd':
            opt_args.setdefault('betas', (0.95, 0.98))
            optimizer = FusedNovoGrad(parameters, **opt_args)
        else:
            assert False and "Invalid optimizer"
            raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
