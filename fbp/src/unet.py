import torch
import torch.nn as nn

from fbp.src.utils import output_shape_conv2d_layers, padding_transpose_conv2d_layers
from fbp.src.initialize_weights import init_weights


class AttentionGate(nn.Module):
    def __init__(self,
                 in_channels,
                 gating_channels,
                 inter_channels):
        super().__init__()
        self.W_x = nn.Conv2d(in_channels=in_channels,
                             out_channels=inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=gating_channels,
                             out_channels=inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=False)
        self.psi = nn.Conv2d(in_channels=inter_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        psi = self.relu(x1 + g1)
        psi = self.sigmoid(self.psi(psi))

        return x * psi


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 filters_root: int = 8,
                 kernel_size: tuple[int, int] = (3, 7),
                 stride: tuple[int, int] = (3, 3),
                 depth: int = 5,
                 activation=torch.relu,
                 output_activation=torch.nn.Softmax(dim=1),
                 input_shape: tuple = (32, 2048),
                 drop_rate: float = 0.0,
                 skip_connections: bool = True,
                 attention: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.activation = activation
        self.output_activation = output_activation
        self.skip_connections = skip_connections
        self.input_shape = input_shape
        self.attention = attention
        self.dropout = nn.Dropout(p=drop_rate)

        self.inc = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.filters_root,
                             kernel_size=self.kernel_size,
                             padding="same",
                             bias=False)
        self.in_bn = nn.BatchNorm2d(num_features=self.filters_root, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        last_filters = self.filters_root
        down_shapes = {-1: self.input_shape}
        for i in range(self.depth):
            filters = int(2 ** i * self.filters_root)
            conv_same = nn.Conv2d(in_channels=last_filters,
                                  out_channels=filters,
                                  kernel_size=self.kernel_size,
                                  padding="same",
                                  bias=False)
            bn1 = nn.BatchNorm2d(num_features=filters,
                                 eps=1e-3)

            conv_same2 = nn.Conv2d(in_channels=filters,
                                   out_channels=filters,
                                   kernel_size=self.kernel_size,
                                   padding="same",
                                   bias=False)
            bn12 = nn.BatchNorm2d(num_features=filters,
                                  eps=1e-3)

            last_filters = filters
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                down_shapes[i] = output_shape_conv2d_layers(input_shape=down_shapes[i - 1],
                                                            padding=(1, 1),
                                                            kernel_size=self.kernel_size,
                                                            stride=self.stride)
                conv_down = nn.Conv2d(in_channels=filters,
                                      out_channels=filters,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=(1, 1),
                                      bias=False)
                bn2 = nn.BatchNorm2d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([
                conv_same, bn1,
                conv_same2, bn12,
                conv_down, bn2
            ]))

        for i in range(self.depth - 2, -1, -1):
            filters = int(2 ** i * self.filters_root)

            padding = padding_transpose_conv2d_layers(
                input_shape=down_shapes[i],
                output_shape=down_shapes[i - 1],
                kernel_size=self.kernel_size,
                stride=self.stride
            )

            conv_up = nn.ConvTranspose2d(in_channels=last_filters,
                                         out_channels=filters,
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding=padding,
                                         bias=False)
            bn1 = nn.BatchNorm2d(filters,
                                 eps=1e-3)

            in_channels_conv_same = filters * 2 if self.skip_connections else filters
            conv_same = nn.Conv2d(in_channels=in_channels_conv_same,
                                  out_channels=filters,
                                  kernel_size=self.kernel_size,
                                  padding="same",
                                  bias=False)
            bn2 = nn.BatchNorm2d(filters,
                                 eps=1e-3)

            last_filters = filters
            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))
            self.attention_gates.append(AttentionGate(filters, last_filters, filters // 2))

        self.out = nn.Conv2d(in_channels=last_filters,
                             out_channels=self.out_channels,
                             kernel_size=(1, 1),
                             padding="same")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.activation(self.in_bn(self.inc(x)))
        x = self.dropout(x)
        skip_connections = []
        shapes = {-1: x.shape}

        for i, (conv_same, bn1, conv_same2, bn12, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))
            x = self.activation(bn12(conv_same2(x)))

            if conv_down is not None:
                skip_connections.append(x)
                shapes[i] = x.shape
                x = self.activation(bn2(conv_down(x)))

            x = self.dropout(x)

        for (conv_up, bn1, conv_same, bn2), att_gate, skip in zip(self.up_branch, self.attention_gates,
                                                                  reversed(skip_connections)):
            x = self.activation(bn1(conv_up(x)))
            x = self.dropout(x)
            x = self.crop(x, skip)
            if self.skip_connections:
                if self.attention:
                    skip = att_gate(skip, x)
                x = torch.cat([skip, x], dim=1)
            x = self.activation(bn2(conv_same(x)))
            x = self.dropout(x)

        x = self.output_activation(self.out(x)) if self.output_activation else self.out(x)
        return x

    @staticmethod
    def crop(tensor, target_tensor):
        _, _, h, w = target_tensor.size()
        return tensor[:, :, :h, :w]

    def get_model_args(self):
        """
        Returning dictionary with all input arguments for U-Net
        :return:
        """
        model_args = dict()
        model_args["in_channels"] = self.in_channels
        model_args["out_channels"] = self.out_channels
        model_args["filter_root"] = self.filters_root
        model_args["kernel_size"] = self.kernel_size
        model_args["stride"] = self.stride
        model_args["depth"] = self.depth
        model_args["activation"] = self.activation
        model_args["output_activation"] = self.output_activation
        model_args["input_shape"] = self.input_shape
        model_args["drop_rate"] = self.drop_rate
        model_args["skip_connections"] = self.skip_connections
        model_args["attention"] = self.attention

        return model_args
