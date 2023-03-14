import torch
import torch.nn as nn
import torch.nn.functional as F
from Multilayer_Perceptron import MLP


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # AvgPool1d的作用是对输入的数据进行平均池化，kernel_size是池化窗口的大小，stride是池化窗口的移动步长，padding是输入的每一条数据两端补0的长度。
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # front和end分别是x的第一个和最后一个元素，repeat是将front和end重复，重复的次数是(kernel_size - 1) // 2，即前后各补(kernel_size - 1) // 2个元素。这样做的目的是为了保证x的长度不变，因为AvgPool1d会将x的长度缩短。
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)) # permute是将x的维度进行转置，即将x的维度从(batch_size, seq_len, channels)转置为(batch_size, channels, seq_len)。
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x) # moving_avg用于计算x的移动平均，即trend，能够突出时间序列的趋势，但是会丢失季节性信息，因此需要将trend和seasonal分开。
        # moving_avg对于
        res = x - moving_mean 
        return res, moving_mean # res是seasonal，moving_mean是trend


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    progressive decomposition architecture——渐进分解架构,将时间序列分解为trend和seasonal两部分，trend用于突出时间序列的趋势，seasonal用于突出时间序列的季节性。
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        
        """add a multilayer perceptron to the encoder layer"""
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, d_model*2),
        #     nn.ReLU(),
        #     nn.Linear(d_model*2, d_model)
        # )
        self.mlp = MLP(d_model, d_model*2, d_model, dropout, activation) #todo 在原来的feed forward层后添加了一个MLP层

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x) # new_x和x相加，相当于将attention的输出和原始输入相加，这样做的目的是为了保留原始输入的信息，防止attention的输出信息过多，导致模型过拟合。
        x, _ = self.decomp1(x)

        y = x #! y是x的一个副本，作为feed forward层的输入
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) 
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        #todo 在这里使用 MLP 对 y 进行加工
        x_mlp = self.mlp(y)
        res, _ = self.decomp2(x + x_mlp) #todo 把 MLP的输出和x相加
        # res, _ = self.decomp2(x + y) #! x + y就是feed forward层的输出
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        # ModuleList是一个可以包含子模块的列表，可以像访问列表一样访问ModuleList中的子模块
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask) # attn_layer可以理解为attention层，attn_mask是attention mask，用于屏蔽掉一些不需要的attention
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer(nn.Module): # nn.Module是所有神经网络模块的基类
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model # d_ff是feed forward的维度，d_model是模型的维度，d_ff默认为4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        #! 下面的conv1和conv2叠加起来就是一个feed forward层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False) # 
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)

        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu #?这个激活函数也属于feed forward层的一部分
        """add a multilayer perceptron to the decoder layer"""
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model)
        # )
        self.mlp = MLP(d_model, d_model*2, d_model, dropout, activation)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x #! y是x的一个副本，作为feed forward层的输入
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        #todo 在这里使用 MLP 对 x 进行加工
        x_mlp = self.mlp(y)
        x, trend3 = self.decomp3(x_mlp + x) #todo 把 MLP的输出和x相加
        # x, trend3 = self.decomp3(x + y) #! x + y就是feed forward层的输出

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2) # projection是一个卷积层，用于将维度从d_model变为c_out
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
