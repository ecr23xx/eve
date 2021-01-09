# This code is adpated from https://github.com/Lyken17/pytorch-OpCounter
import torch
import logging
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

import MinkowskiEngine as ME


def zero_ops(m, x, y):
    m.total_ops += torch.Tensor([int(0)])


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor(
        [*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logger.warning(
            "mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_minkowski_convNd(m, x: (ME.SparseTensor,), y: ME.SparseTensor):
    if m.use_mm:
        total_mul = m.in_channels
        total_add = m.in_channels - 1
        total_add += 1 if m.bias is not None else 0
        num_elements = y.F.numel()
        total_ops = (total_mul + total_add) * num_elements
        m.total_ops += torch.Tensor([int(total_ops)])
    else:
        coords_man = x[0].coords_man
        if m.is_transpose:
            try:
                kernel_map_size = coords_man.get_kernel_map(
                    x[0].tensor_stride,
                    y.tensor_stride,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    dilation=m.dilation,
                    region_type=m.region_type_,
                    is_transpose=m.is_transpose).size(0)
            except Exception:
                kernel_map_size = coords_man.get_kernel_map(
                    y.tensor_stride,
                    x[0].tensor_stride,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    dilation=m.dilation,
                    region_type=m.region_type_,
                    is_transpose=False).size(0)
        else:
            kernel_map_size = coords_man.get_kernel_map(
                x[0].tensor_stride,
                y.tensor_stride,
                kernel_size=m.kernel_size,
                stride=m.stride,
                dilation=m.dilation,
                region_type=m.region_type_,
                is_transpose=m.is_transpose).size(0)

        # kernel map size x Cin x Cout
        total_ops = kernel_map_size * m.in_channels * m.out_channels
        m.total_ops += torch.Tensor([int(total_ops)])


def profiler(model, inputs, custom_ops=None):
    logger = logging.getLogger('eve.' + __name__)

    register_hooks = {
        nn.Conv1d: count_convNd,
        nn.Conv2d: count_convNd,
        nn.Conv3d: count_convNd,
        nn.ConvTranspose1d: count_convNd,
        nn.ConvTranspose2d: count_convNd,
        nn.ConvTranspose3d: count_convNd,

        nn.BatchNorm1d: count_bn,
        nn.BatchNorm2d: count_bn,
        nn.BatchNorm3d: count_bn,

        nn.ReLU: zero_ops,
        nn.ReLU6: zero_ops,
        nn.LeakyReLU: count_relu,

        nn.MaxPool1d: zero_ops,
        nn.MaxPool2d: zero_ops,
        nn.MaxPool3d: zero_ops,
        nn.AdaptiveMaxPool1d: zero_ops,
        nn.AdaptiveMaxPool2d: zero_ops,
        nn.AdaptiveMaxPool3d: zero_ops,

        nn.AvgPool1d: count_avgpool,
        nn.AvgPool2d: count_avgpool,
        nn.AvgPool3d: count_avgpool,
        nn.AdaptiveAvgPool1d: count_adap_avgpool,
        nn.AdaptiveAvgPool2d: count_adap_avgpool,
        nn.AdaptiveAvgPool3d: count_adap_avgpool,

        nn.Linear: count_linear,
        nn.Dropout: zero_ops,
        nn.Identity: zero_ops,

        nn.Upsample: count_upsample,
        nn.UpsamplingBilinear2d: count_upsample,
        nn.UpsamplingNearest2d: count_upsample,

        nn.Sequential: zero_ops,

        ME.MinkowskiConvolution: count_minkowski_convNd,
        ME.MinkowskiConvolutionTranspose: count_minkowski_convNd,
    }

    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning("Either .total_ops or .total_params is already defined in %s. "
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in register_hooks:
            fn = register_hooks[m_type]
            # logger.info("Register FLOP counter for module {}".format(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
        else:
            logger.warning(
                "Not implemented counting method for {}".format(m))

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(**inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

    return total_ops, total_params
