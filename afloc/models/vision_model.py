from numpy.lib.function_base import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from omegaconf import OmegaConf
import torch.nn.functional as F


def hook_fn1(module, input, output):
    module.hook_fn1_output = output

def hook_fn2(module, input, output):
    module.hook_fn2_output = output


class ImageEncoder(nn.Module):
    """Image encoder of AFLOC."""

    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg
        self.output_dim = cfg.model.text.embedding_dim
        
        # create model
        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        # hook for intermediate features
        if "densenet" in self.cfg.model.vision.model_name:
            if self.cfg.model.vision.mode == 1:
                self.hook_fn1_module = self.model.features.denseblock4.denselayer1.relu1
                self.hook_fn2_module = self.model.features.denseblock3.denselayer1.relu1
                self.hook_fn1_module.register_forward_hook(hook_fn1)
                self.hook_fn2_module.register_forward_hook(hook_fn2)
                self.interm_feature_dim = 512
            elif self.cfg.model.vision.mode == 2:
                self.hook_fn1_module = self.model.features.denseblock3.denselayer1.relu1
                self.hook_fn2_module = self.model.features.denseblock2.denselayer1.relu1
                self.hook_fn1_module.register_forward_hook(hook_fn1)
                self.hook_fn2_module.register_forward_hook(hook_fn2)
                self.interm_feature_dim = 256
            elif self.cfg.model.vision.mode == 3:
                self.hook_fn1_module = self.model.features.denseblock2.denselayer1.relu1
                self.hook_fn2_module = self.model.features.denseblock1.denselayer1.relu1
                self.hook_fn1_module.register_forward_hook(hook_fn1)
                self.hook_fn2_module.register_forward_hook(hook_fn2)
                self.interm_feature_dim = 128

        # embedders for global and local features
        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        if self.cfg.model.afloc.use_local_word_loss:
            self.local2_embedder = nn.Conv2d(
                self.interm_feature_dim//2,
                self.output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        
        if self.cfg.model.afloc.use_ce_loss:
            self.local_f_embedder = nn.Conv2d(
                self.interm_feature_dim * 2,
                self.output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # freeze the CNN model
        if cfg.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        """
        Forward pass.
        
        Inputs:
            x: input image tensor
            get_local: whether to return local features

        Returns:
            global_ft (torch.Tensor): global features
            local_ft (torch.Tensor): local features
            local_ft2 (torch.Tensor): local features 2
            local_ftf (torch.Tensor): local features final
        """

        # --> fixed-size input: batch x 3 x 299 x 299
        if "resnet" in self.cfg.model.vision.model_name or "resnext" in self.cfg.model.vision.model_name:
            global_ft, local_ft, local_ft2, local_ftf = self.resnet_forward(x, extract_features=True)
        elif "densenet" in self.cfg.model.vision.model_name:
            global_ft, local_ft, local_ft2 = self.densenet_forward(x, extract_features=True)

        if get_local:
            return global_ft, local_ft, local_ft2, local_ftf
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features, local_features2, local_features_final):
        """
        Map image features to the same dimension of text embeddings.

        Inputs:
            global_features (torch.Tensor): global features
            local_features (torch.Tensor): local features
            local_features2 (torch.Tensor): local features 2
            local_features_final (torch.Tensor): local features final

        Returns:
            global_emb (torch.Tensor): global embeddings
            local_emb (torch.Tensor): local embeddings
            local_emb2 (torch.Tensor): local embeddings 2
            local_embf (torch.Tensor): local embeddings final
        """

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)
        local_emb2 = None
        local_embf = None
        if self.cfg.model.afloc.use_local_word_loss:
            local_emb2 = self.local2_embedder(local_features2)
        if self.cfg.model.afloc.use_ce_loss:
            local_embf = self.local_f_embedder(local_features_final)

        return global_emb, local_emb, local_emb2, local_embf

    def resnet_forward(self, x, extract_features=False):
        """
        Forward pass for ResNet models.

        Inputs:
            x (torch.Tensor): batch of images (batch_size, channel, height, width)
            extract_features (bool): whether to extract features

        Returns:
            x (torch.Tensor): global features (batch_size, 512)
            local_features (torch.Tensor): local features (batch_size, 256, 19, 19)
            local_features2 (torch.Tensor): local features 2 (batch_size, 128, 38, 38)
            local_features_final (torch.Tensor): local features final (batch_size, 512, 10, 10)
        """

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        local_features2 = x
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)
        local_features_final = x

        x = self.pool(x) # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1) # (batch_size, 512)

        return x, local_features, local_features2, local_features_final

    def densenet_forward(self, x, extract_features=False):
        """
        Forward pass for DenseNet models.

        Inputs:
            x (torch.Tensor): batch of images (batch_size, 3, height, width)
            extract_features (bool): whether to extract features

        Returns:
            x (torch.Tensor): global features (batch_size, 512)
            local_features (torch.Tensor): local features (batch_size, 256, 19, 19)
            local_features2 (torch.Tensor): local features 2 (batch_size, 128, 38, 38)
        """

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        local_features = self.hook_fn1_module.hook_fn1_output
        local_features2 = self.hook_fn2_module.hook_fn2_output

        return x, local_features, local_features2

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

