"""
ResNet-50 CNN 检测模型
接口完全兼容 model.py (forward_frame / forward)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CNNBaseline(nn.Module):
    """ResNet-50 backbone + 2-class classification head"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)

        # 替换最后全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # 注册 hook 辅助变量
        self.gradients = None
        self.activations = None

    def _hook_fn_forward(self, module, input, output):
        self.activations = output

    def _hook_fn_backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def register_gradcam_hooks(self, target_layer_name: str = "layer4"):
        """注册 Grad-CAM 所需的 forward/backward hooks"""
        target = dict(self.backbone.named_modules())[target_layer_name]
        target.register_forward_hook(self._hook_fn_forward)
        target.register_full_backward_hook(self._hook_fn_backward)

    def forward_frame(self, x: torch.Tensor):
        """输入单帧图像 [B, C, H, W]，返回 logits"""
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        """输入视频时序 [B, T, C, H, W]，返回视频级 logits"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        logits = self.forward_frame(x)
        logits = logits.view(B, T, -1).mean(dim=1)
        return logits

    def forward_features(self, x: torch.Tensor):
        """返回 layer4 的特征图 (用于 Grad-CAM)"""
        # ResNet 的前几层提取，不包括最后的 avgpool+fc
        features = {}
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features['layer1'] = self.backbone.layer1(x)
        features['layer2'] = self.backbone.layer2(features['layer1'])
        features['layer3'] = self.backbone.layer3(features['layer2'])
        features['layer4'] = self.backbone.layer4(features['layer3'])
        return features


def create_model(variant: str = "resnet50", num_classes: int = 2, pretrained: str = None):
    """兼容 create_model 接口"""
    use_pretrained = pretrained is not None if pretrained else True
    model = CNNBaseline(num_classes=num_classes, pretrained=use_pretrained)
    if pretrained and pretrained != "imagenet":
        ckpt = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    return model
