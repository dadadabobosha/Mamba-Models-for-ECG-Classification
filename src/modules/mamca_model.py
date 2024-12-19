import torch
import torch.nn as nn
from functools import partial
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import GenerationMixin
from .selective_SSM import MixerModel, _init_weights


class denosing_unit(nn.Module):
    def __init__(self, block, layers, in_channel=1, out_channel=16):
        super(denosing_unit, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layers = nn.Sequential(
            block(in_channel, out_channel),
            *[block(out_channel, out_channel) for _ in range(layers - 1)],
        )

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AdaptiveClassifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model if i == 0 else 64, 64,
                          kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for i in range(4)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(32)
        self.fc = nn.Sequential(
            nn.Linear(64 * 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, d_model, seq_len]
        for conv in self.conv_layers:
            x = conv(x)
        x = self.global_pool(x)  # -> [batch, 64, 32]
        x = x.flatten(1)  # -> [batch, 64*32]
        return self.fc(x)


class MAMCA(nn.Module, GenerationMixin):
    def __init__(
            self,
            config: MambaConfig,
            length=1000,
            num_claasses=2,
            device=None,
            dtype=None,
            fused_add_norm=False,
    ):
        super().__init__()
        self.config = config
        self.fused_add_norm = False

        self.dropout = nn.Dropout(0.5)
        self.spec_augment = nn.Sequential(
            nn.Dropout(0.1),
            nn.Dropout2d(0.1)  # Channel dropout
        )

        d_model = config.d_model
        n_layer = config.n_layer
        ssm_cfg = config.ssm_cfg
        residual_in_fp32 = config.residual_in_fp32
        factory_kwargs = {"device": device, "dtype": dtype}

        # Backbone
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            rms_norm=False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        # Denosing unit
        self.denosing = denosing_unit(
            BasicBlock, 2, in_channel=1, out_channel=config.d_model
        )

        # New adaptive classifier that works for any sequence length
        self.classifier = AdaptiveClassifier(config.d_model, num_claasses)

    def forward(self, hidden_states, inference_params=None):
        if self.training:
            hidden_states = self.spec_augment(hidden_states)

        hidden_states = self.denosing(hidden_states)
        hidden_states = self.backbone(hidden_states, inference_params=inference_params)
        # No need to flatten since AdaptiveClassifier handles the dimension reduction
        return self.classifier(hidden_states.transpose(1, 2))  # Transpose for Conv1d


def get_model(input_length, num_classes, device="cuda"):
    config = MambaConfig()
    config.d_model = 16
    config.n_layer = 1
    config.ssm_cfg = {
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    }

    model = MAMCA(
        config=config,
        length=input_length,  # Note: length parameter is no longer used in classifier
        num_claasses=num_classes,
        device=device,
        fused_add_norm=False,
    )
    return model.to(device)


def _init_weights(module, n_layer):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
