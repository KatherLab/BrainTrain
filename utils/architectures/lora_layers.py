import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Original Conv3D LoRA (UNCHANGED in behavior)
# ============================================================
class Conv3DWithLoRA(nn.Module):
    def __init__(self, original_conv, rank, alpha):
        super().__init__()

        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # freeze original conv
        for p in self.original_conv.parameters():
            p.requires_grad = False

        weight_shape = original_conv.weight.shape
        self.out_features = weight_shape[0]
        self.in_features = (
            weight_shape[1]
            * weight_shape[2]
            * weight_shape[3]
            * weight_shape[4]
        )

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # original conv
        base = self.original_conv(x)

        # LoRA weight
        delta_w = (self.lora_B @ self.lora_A).view(
            self.original_conv.weight.shape
        ) * self.scaling

        # LoRA conv
        lora_out = F.conv3d(
            x,
            delta_w,
            bias=None,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups,
        )

        return base + lora_out


def apply_lora_to_conv3d(conv_module, rank, alpha):
    return Conv3DWithLoRA(conv_module, rank, alpha)


# ============================================================
# Apply LoRA to model (ABLATION-COMPATIBLE)
# ============================================================
def apply_lora_to_model(
    model,
    rank=16,
    alpha=64,
    target_modules=None,
):
    """
    Apply LoRA to Conv3D layers whose names match
    any string in target_modules.

    Example target_modules:
      ['feature_extractor.conv_']      # all conv layers
      ['feature_extractor.conv_5']     # late only
      ['feature_extractor.conv_0']     # early only
      ['feature_extractor.conv_4']     # mid only
    """

    if target_modules is None:
        target_modules = ['conv_']

    replaced = 0

    for name, module in list(model.named_modules()):

        if not isinstance(module, nn.Conv3d):
            continue

        # string-based matching (UNCHANGED conceptually)
        if not any(t in name for t in target_modules):
            continue

        # find parent
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        child_name = parts[-1]

        print(f"[LoRA] Applying to: {name}")
        setattr(
            parent,
            child_name,
            apply_lora_to_conv3d(module, rank, alpha)
        )

        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            f"No Conv3D layers matched LORA_TARGET_MODULES={target_modules}"
        )

    print(
        f"[LoRA] Total layers adapted: {replaced} | "
        f"Targets: {target_modules}"
    )

    return model
