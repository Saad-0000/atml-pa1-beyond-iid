import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50Wrapper(nn.Module):

    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        base = models.resnet50(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.feature_dim = 2048

        self.normalize = weights.transforms()
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ViTB16Wrapper(nn.Module):

    def __init__(self):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.base = models.vit_b_16(weights=weights)
        self.feature_dim = 768
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.base._process_input(x)
        cls_token = self.base.class_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.base.encoder(x)
        return x[:, 0]


class OpenCLIPWrapper(nn.Module):

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        self.feature_dim = 512
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(x)
        return F.normalize(features, p=2, dim=-1)

    def get_text_prompts(
        self, classes: list, device: str = "cpu"
    ) -> torch.Tensor:
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        prompts = [f"a photo of a {c}." for c in classes]
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_embeds = self.model.encode_text(tokens)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        return text_embeds