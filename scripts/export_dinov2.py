"""Export DINOv2-small as an ONNX image embedder for FastLabel's cluster mode.

Input  `images`: float32 [N, 3, 224, 224], RGB, ImageNet-normalized.
Output `embeddings`: float32 [N, 384], the normalized CLS token.

Run with any Python that has torch (weights are downloaded once by torch.hub):
    python scripts/export_dinov2.py [out.onnx]
"""

import sys

import torch

out = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vits14.onnx"


class ClsToken(torch.nn.Module):
    """Images in, normalized CLS token out (no optional `masks` input)."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    def forward(self, images):
        return self.backbone.forward_features(images)["x_norm_clstoken"]


model = ClsToken().eval()
dummy = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy,),
        out,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
print(f"wrote {out}")
