import torch
import spconv.pytorch as spconv

def calculate_gemm_flops(x, batch_dict, indice_key, inchannel, outchannel):
    pair_fwd = x.indice_dict[indice_key].pair_fwd
    cur_flops = 2 * (pair_fwd > -1).sum() * inchannel * outchannel - pair_fwd.shape[1]
    return cur_flops

def calculate_sparse_conv_parameters(x,indice_key, inchannel, outchannel, use_bias=True):
    """
    计算稀疏卷积的参数量。

    参数:
    w (int): 卷积核的宽度
    h (int): 卷积核的高度
    d (int): 卷积核的深度
    inchannel (int): 输入通道数
    outchannel (int): 输出通道数
    use_bias (bool): 是否使用偏置项

    返回:
    int: 稀疏卷积的参数量
    """
    
    kernel_params = 3*3*3 * inchannel * outchannel
  
    # 如果使用偏置，则每个输出通道有一个偏置参数
    bias_params = outchannel if use_bias else 0

    # 总参数量是卷积核参数量加上偏置参数量（如果有的话）
    total_params = kernel_params + bias_params

    return total_params