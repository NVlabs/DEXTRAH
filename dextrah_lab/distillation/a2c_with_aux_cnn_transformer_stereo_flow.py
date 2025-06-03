import math

from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.common.layers.recurrent import GRUWithDones, LSTMWithDones
from rl_games.common.layers.value import TwoHotEncodedValue, DefaultValue
from rl_games.algos_torch.running_mean_std import RunningMeanStd


def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)


class NetworkBuilder:
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
            self.activations_factory.register_builder('swish', lambda **kwargs : nn.SiLU(**kwargs))
            self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())

            self.init_factory = object_factory.ObjectFactory()
            #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
            self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
            self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
            self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
            self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(torch_ext.variance_scaling_initializer,**kwargs))
            self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
            self.init_factory.register_builder('kaiming_normal', lambda **kwargs : _create_initializer(nn.init.kaiming_normal_,**kwargs))
            self.init_factory.register_builder('orthogonal', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('default', lambda **kwargs : nn.Identity() )

        def is_separate_critic(self):
            return False

        def get_value_layer(self):
            return self.value

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _calc_input_size(self, input_shape,cnn_layers=None):
            if cnn_layers is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _noisy_dense(self, inputs, units):
            return layers.NoisyFactorizedLinear(inputs, units)

        def _build_rnn(self, name, input, units, layers):
            if name == 'identity':
                return torch_ext.IdentityRNN(input, units)
            if name == 'lstm':
                return LSTMWithDones(input_size=input, hidden_size=units, num_layers=layers)
            if name == 'gru':
                return GRUWithDones(input_size=input, hidden_size=units, num_layers=layers)

        def _build_sequential_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func,
        norm_only_first_layer=False, 
        norm_func_name = None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))

                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                   need_norm = False 
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_only_first_layer=False,
        norm_func_name = None,
        d2rl=False):
            if d2rl:
                act_layers = [self.activations_factory.create(activation) for i in range(len(units))]
                return D2RLNet(input_size, units, act_layers, norm_func_name)
            else:
                return self._build_sequential_mlp(input_size, units, activation, dense_func, norm_func_name = None,)

        def _build_conv(self, ctype, **kwargs):
            print('conv_name:', ctype)

            if ctype == 'conv2d':
                return self._build_cnn2d(**kwargs)
            if ctype == 'coord_conv2d':
                return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
            if ctype == 'conv1d':
                return self._build_cnn1d(**kwargs)

        def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None):
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(conv_func(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                conv_func=torch.nn.Conv2d
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print('conv1d input shape:', input_shape)
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(torch.nn.Conv1d(in_channels, conv['filters'], conv['kernel_size'], conv['strides'], conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_value_layer(self, input_size, output_size, value_type='legacy'):
            if value_type == 'legacy':
                return torch.nn.Linear(input_size, output_size)
            if value_type == 'default':
                return DefaultValue(input_size, output_size)            
            if value_type == 'twohot_encoded':
                return TwoHotEncodedValue(input_size, output_size)

            raise ValueError('value type is not "default", "legacy" or "two_hot_encoded"')


CNN_OUT_FEATURES = 32


class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, inference=False):
        # x is (batch_size, seq_len, latent_dim)

        attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)

        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False, is_causal=True, attn_mask=attn_mask)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, 
        context_len, latent_dim=128, 
        num_head=4, num_layer=4, dropout_rate=0.1,
        channels_t=128
    ) -> None:
        super().__init__()
        self.channels_t = channels_t
        self.input_dim = input_dim + channels_t
        self.output_dim = output_dim
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.weight_pos_embed = nn.Embedding(context_len, latent_dim)
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )

    def gen_t_embedding(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels_t % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, t):
        # get time embedding
        t = self.gen_t_embedding(t)
        t = t[:, None, :].repeat(1, self.context_len, 1)
        x = torch.cat([x, t], dim=-1)
        # input is (batch_size, seq_len, input_dim)
        x = self.input_layer(x)
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.attention_blocks(x)

        # # take the last token
        # x = x[:, -1, :]
        x = self.output_layer(x)

        return x


CNN_OUT_FEATURES = 32
AUX_LATENT_DIM = 64
def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function to compute the output size of a convolution layer.
    
    h_w: Tuple[int, int] - height and width of the input
    kernel_size: int or Tuple[int, int] - size of the convolution kernel
    stride: int or Tuple[int, int] - stride of the convolution
    pad: int or Tuple[int, int] - padding
    dilation: int or Tuple[int, int] - dilation rate
    """
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size
    
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride
    
    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad
    
    h = (h_w[0] + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    w = (h_w[1] + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    return h, w

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        
        # Create a grid of coordinates (height, width)
        pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height))
        self.register_buffer("pos_x", pos_x.reshape(height * width))
        self.register_buffer("pos_y", pos_y.reshape(height * width))

    def forward(self, x):
        # Reshape input to (batch_size, num_channels, height * width)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        
        # Apply softmax over the spatial dimensions (height * width)
        softmax_attention = F.softmax(x, dim=-1)  # [batch_size, num_channels, height * width]
        
        # Compute the expected coordinates for x and y
        exp_x = torch.sum(softmax_attention * self.pos_x, dim=-1)  # [batch_size, num_channels]
        exp_y = torch.sum(softmax_attention * self.pos_y, dim=-1)  # [batch_size, num_channels]
        
        # Concatenate x and y expected coordinates for each channel
        spatial_softmax_output = torch.cat([exp_x, exp_y], dim=-1)  # [batch_size, num_channels * 2]
        
        return spatial_softmax_output
class CustomCNN(nn.Module):
    def __init__(self, input_height, input_width, device, depth=True):
        super().__init__()
        self.device = device
        num_channel = 1 if depth else 3
        # num_channel *= 2
        
        # Initial input dimensions
        h, w = input_height, input_width
        
        # Layer 1
        h, w = conv_output_size((h, w), kernel_size=6, stride=2)
        layer1_norm_shape = [16, h, w]
        
        # Layer 2
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer2_norm_shape = [32, h, w]
        
        # Layer 3
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer3_norm_shape = [64, h, w]
        
        # Layer 4
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer4_norm_shape = [128, h, w]
        
        # CNN definition
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer1_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer2_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer3_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            SpatialSoftmax(h, w),
        )
        
        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(128 * 2, CNN_OUT_FEATURES)
        )

        self.resnet18_mean = torch.tensor([0.485, 0.0456, 0.0406], device=self.device)
        self.resnet18_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        self.resnet_transform = transforms.Normalize(self.resnet18_mean, self.resnet18_std)

    def forward(self, x):
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128*2))
        return out


class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape') 
            input_shape = (input_shape[0] + 2*CNN_OUT_FEATURES,)
            # input is observations + actions to be denoised
            input_shape = (input_shape[0] + actions_num,)
            self.num_obs = input_shape[0]
            self.num_acts = actions_num
            self.num_envs = kwargs.pop('num_envs', 1)

            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'],
                    'input_shape' : input_shape,
                    'convs' :self.cnn['convs'],
                    'activation' : self.cnn['activation'],
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

                if self.is_aux:
                    mlp_args = {
                        'input_size': self.units[-1] + input_shape[0],
                        'units': self.aux_units,
                        'activation': self.aux_activation,
                        'norm_func_name': self.aux_network.get('normalization', None),
                        'dense_func': torch.nn.Linear,
                        'd2rl': self.aux_is_d2rl,
                        'norm_only_first_layer': self.aux_norm_only_first_layer
                    }

                    self.aux_mlp = self._build_mlp(**mlp_args)
                    self.aux_mlp = torch.compile(self.aux_mlp)
                    self.aux_networks = nn.ModuleDict()

                    for output_name in self.aux_outputs:
                        aux_out_size = self.aux_heads[output_name]["size"]
                        self.aux_networks[output_name] = nn.Sequential(
                            nn.Linear(self.aux_units[-1], aux_out_size),
                            self.activations_factory.create(self.aux_out_activation)
                        )
                        # assert len(input_shape[output_name]) == 1
                        # aux_out_size = input_shape[output_name][0]
                        # self.aux_networks[output_name] = nn.Sequential(
                        #     nn.Linear(self.aux_units[-1], aux_out_size),
                        #     self.activations_factory.create(self.aux_out_activation)
                        # )
            else:
                if self.is_aux:
                    mlp_args = {
                        # 'input_size': self.rnn_units + in_mlp_shape,
                        'input_size': AUX_LATENT_DIM,
                        'units': self.aux_units,
                        'activation': self.aux_activation,
                        'norm_func_name': self.aux_network.get('normalization', None),
                        'dense_func': torch.nn.Linear,
                        'd2rl': self.aux_is_d2rl,
                        'norm_only_first_layer': self.aux_norm_only_first_layer
                    }
                    self.aux_mlp = self._build_mlp(**mlp_args)

                    self.aux_networks = nn.ModuleDict()

                    for output_name in self.aux_outputs:
                        # assert len(input_shape[output_name]) == 1
                        # aux_out_size = input_shape[output_name][0]
                        aux_out_size = self.aux_heads[output_name]["size"]
                        self.aux_networks[output_name] = nn.Sequential(
                            nn.Linear(self.aux_units[-1], aux_out_size),
                            self.activations_factory.create(self.aux_out_activation)
                        )

            self.img_height = int(120 * 2)
            self.img_width = int(160 * 2)
            self.use_depth = False
            self.feature_extractor = CustomCNN(
                input_height=self.img_height,
                input_width=self.img_width,
                device="cuda", depth=self.use_depth
            )
            self.feature_extractor = torch.compile(self.feature_extractor)
            if self.is_aux:
                self.transformer_output_size = self.num_acts + AUX_LATENT_DIM # latent dim for aux
            else:
                self.transformer_output_size = self.num_acts

            self.transformer = Transformer(
                self.num_obs,
                self.transformer_output_size + 1,
                self.transformer_context_length,
            )
            self.init_tensors()
            # self.feature_extractor_right = CustomCNN(
            #     input_height=self.img_height,
            #     input_width=self.img_width,
            #     device="cuda", depth=self.use_depth
            # )
            mlp_args = {
                'input_size' : in_mlp_shape + input_shape[0],
                'units' : self.units,
                'activation' : self.activation,
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            self.actor_mlp = torch.compile(self.actor_mlp)
            self.running_mean_img = True
            num_channels = 1 if self.use_depth else 3
            self.running_mean_std = RunningMeanStd(
                (num_channels, self.img_height, self.img_width)
            )
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                #     cnn_init(m.weight)
                #     if getattr(m, "bias", None) is not None:
                #         torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

        def init_tensors(self):
            self.observation_history = torch.zeros(
                (self.num_envs, self.transformer_context_length, self.num_obs),
                dtype=torch.float,
            )
            self.action_history = torch.zeros(
                (self.num_envs, self.transformer_context_length, self.num_acts),
                dtype=torch.float,
            )
            self.history_mask = torch.zeros(
                (self.num_envs, self.transformer_context_length),
                dtype=torch.bool,
            )

        def reset_idx(self, env_ids):
            self.observation_history[env_ids] = 0
            self.action_history[env_ids] = 0
            self.history_mask[env_ids] = 0

        def update_observation_history(self, obs):
            # call in post_physics_step
            self.observation_history = torch.roll(self.observation_history, shifts=-1, dims=1).detach()
            self.history_mask = torch.roll(self.history_mask, shifts=-1, dims=1)

            self.observation_history[:, -1, :] = obs
            self.history_mask[:, -1] = 1

        def update_action_history(self, actions):
            # call in pre_physics_step
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = actions

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            if self.observation_history.device != obs.device:
                self.observation_history = self.observation_history.to(obs.device)
                self.action_history = self.action_history.to(obs.device)
                self.history_mask = self.history_mask.to(obs.device)

            noised_actions = obs_dict['noised_actions']
            t = obs_dict['time']

            if "img_left" in obs_dict:
                img_left = obs_dict['img_left'] - torch.mean(
                    obs_dict["img_left"], dim=(2, 3), keepdim=True
                )
                img_right = obs_dict['img_right'] - torch.mean(
                    obs_dict["img_right"], dim=(2, 3), keepdim=True
                )

                combined_embeds = self.feature_extractor(
                    torch.cat([img_left, img_right], dim=0)
                )

                img_features_left, img_features_right = combined_embeds.chunk(
                    2, dim=0
                )
                # img_features_left = self.feature_extractor(
                #     img_left
                # )
                # img_features_right = self.feature_extractor(
                #     img_right
                # )
                # imgs = torch.cat([img_left, img_right], dim=1)
                # img_features = self.feature_extractor(imgs)
                obs = torch.cat(
                    [obs, img_features_left, img_features_right, noised_actions],
                    dim=-1
                )
            
            self.update_observation_history(obs)
            masked_obs = self.observation_history * self.history_mask.unsqueeze(-1)
            transformer_out = self.transformer(masked_obs, t)
            mu = transformer_out[:, -1, :self.num_acts]
            value = transformer_out[:, -1, self.num_acts]
            states = None
            sigma = mu * 0.0 + self.sigma_act(self.sigma)
            if self.is_aux:
                self.last_aux_out = {}
                aux_input = self.aux_mlp(
                    transformer_out[:, -1, self.num_acts+1:]
                )
                for output_name in self.aux_outputs:
                    self.last_aux_out[output_name] = self.aux_networks[output_name](aux_input)
                states = (states, self.last_aux_out)

            return mu, sigma, value, states
                # obs = torch.cat(
                #     [obs, img_features],
                #     dim=-1
                # )
            # obs = self.running_mean_std(obs_dict['observations'])
            # TODO: fix this and allow for normalization! 
            # obs = obs_dict["observations"]
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1) 

                concatenated_input = a_out                   

                if self.has_rnn:
                    seq_length = obs_dict.get('seq_length', 1)

                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)

                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

                if self.is_aux:
                    self.last_aux_out = {}
                    aux_input = self.aux_mlp(
                        torch.cat(
                            [a_out, concatenated_input], dim=-1
                        )
                    )
                    for output_name in self.aux_outputs:
                        self.last_aux_out[output_name] = self.aux_networks[output_name](aux_input)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs.clone()
                out = self.actor_cnn(out)
                out = out.flatten(1)

                concatenated_input = out

                if self.has_rnn:
                    seq_length = obs_dict.get('seq_length', 1)

                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out_in = out
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0, 1)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(
                            torch.cat(
                                [out, concatenated_input], dim=-1
                            )
                        )
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)

                if self.is_aux:
                    self.last_aux_out = {}
                    aux_input = self.aux_mlp(
                        torch.cat(
                            [out, concatenated_input], dim=-1
                        )
                    )
                    for output_name in self.aux_outputs:
                        self.last_aux_out[output_name] = self.aux_networks[output_name](aux_input)

                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, (states, self.last_aux_out)
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_transformer = "transformer" in params
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            self.is_aux = 'aux_outputs' in params
            if self.is_aux:
                self.aux_network = params['aux_network']
                self.aux_heads = params["aux_outputs"]
                self.aux_outputs = list(params['aux_outputs'].keys())

                self.aux_units = self.aux_network['mlp']['units']
                self.aux_activation = self.aux_network['mlp']['activation']
                self.aux_out_activation = self.aux_network['mlp']['out_activation']
                # self.aux_initializer = self.aux_network['mlp']['initializer']
                self.aux_is_d2rl = self.aux_network['mlp'].get('d2rl', False)
                self.aux_norm_only_first_layer = self.aux_network['mlp'].get('norm_only_first_layer', False)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_transformer:
                self.transformer_context_length = params['transformer']['context_length']
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
            else:
                self.has_cnn = False

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net
