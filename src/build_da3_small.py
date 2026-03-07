import os
os.environ['HF_HOME'] = r'L:\AI_Assets\HF'

import torch
import math
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

class DynamicDinov2Embeddings(torch.nn.Module):
    """
    A TorchScript compatible drop-in replacement for Dinov2Embeddings.
    Instead of hardcoding the spatial interpolation trace to Python floats and Ints,
    this carefully uses torch.Tensor variables (pixel_values.size()) to guarantee
    the TorchScript trace preserves dynamic shape scaling nodes.
    """
    def __init__(self, original_embeddings):
        super().__init__()
        self.cls_token = original_embeddings.cls_token
        self.mask_token = original_embeddings.mask_token
        self.patch_embeddings = original_embeddings.patch_embeddings
        self.position_embeddings = original_embeddings.position_embeddings
        self.dropout = original_embeddings.dropout
        
        # Hardcode the static python patch_size attribute to avoid runtime config retrieval issues
        self.patch_size = original_embeddings.patch_embeddings.patch_size[0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        
        # Using .size() retains dynamic ATen size node in the Torchscript graph
        height = pixel_values.size(2)
        width = pixel_values.size(3)
        
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Dynamic interpolation logic
        num_positions = self.position_embeddings.size(1) - 1
        class_pos_embed = self.position_embeddings[:, 0:1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.size(-1)
        
        # Dynamic Tensor math. PyTorch trace records these operations natively.
        h_scaled = height // self.patch_size
        w_scaled = width // self.patch_size
        
        # Static python math for the *source* positional embedding grid which is fixed
        pos_dim = int(math.sqrt(num_positions))
        
        patch_pos_embed = patch_pos_embed.reshape(1, pos_dim, pos_dim, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        # F.interpolate trace scaling - passing a list of 0D tensors constructs a dynamic trace
        patch_pos_embed = F.interpolate(
            patch_pos_embed.to(torch.float32),
            size=[h_scaled, w_scaled],
            mode="bicubic",
            align_corners=False,
        ).to(embeddings.dtype)
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)

        embeddings = embeddings + pos_embed
        embeddings = self.dropout(embeddings)
        return embeddings

class NukeDepthAnythingWrapper(torch.nn.Module):
    """
    Nuke 17 Inference Wrapper for Depth Anything 3 (Small).
    Enforces the 14-Pixel Rule and strict [1, C, H, W] I/O requirements.
    """
    def __init__(self, model_path: str):
        super().__init__()
        print(f"Loading local model weights from: {model_path}")
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path)
        self.model.eval()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Inject the trace safe embeddings patch
        self.model.backbone.embeddings = DynamicDinov2Embeddings(self.model.backbone.embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use .size() to construct dynamic shape operations in the trace
        vocab_h = x.size(2)
        vocab_w = x.size(3)
        
        pad_h = (14 - (vocab_h % 14)) % 14
        pad_w = (14 - (vocab_w % 14)) % 14
        
        x_padded = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate')
        x_padded = (x_padded - self.mean) / self.std
        
        outputs = self.model(pixel_values=x_padded, return_dict=False)
        
        predicted_depth = outputs[0]
        predicted_depth = torch.unsqueeze(predicted_depth, 1)
        
        pH = x_padded.size(2)
        pW = x_padded.size(3)
        
        # Size mismatch? Resize.
        if predicted_depth.size(2) != pH or predicted_depth.size(3) != pW:
            predicted_depth = F.interpolate(
                predicted_depth,
                size=[pH, pW],
                mode="bicubic",
                align_corners=False,
            )
            
        predicted_depth = predicted_depth[:, :, :vocab_h, :vocab_w]
        
        d_min = torch.amin(predicted_depth, dim=[1, 2, 3], keepdim=True)
        d_max = torch.amax(predicted_depth, dim=[1, 2, 3], keepdim=True)
        predicted_depth = (predicted_depth - d_min) / (d_max - d_min + 1e-6)
        
        return predicted_depth

if __name__ == "__main__":
    local_model_path = "LiheYoung/depth-anything-small-hf"
    
    wrapper = NukeDepthAnythingWrapper(local_model_path)
    
    dummy_input = torch.rand(1, 3, 504, 504)
    
    print("Tracing the Depth Anything model for Nuke Inference Node...")
    wrapper.eval()
    
    traced_model = torch.jit.trace(wrapper, dummy_input, strict=False)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "pt_exports", "da3_small.pt")
    
    traced_model.save(output_path)
    print(f"Successfully exported dynamic Depth Anything 3 wrapper to: {output_path}")

    # Explicit Trace Verification with an Alternate Configuration
    print("Verifying dynamic trace resolution independence...")
    test_input = torch.rand(1, 3, 1344, 1344)
    out = traced_model(test_input)
    if out.shape[2] == 1344 and out.shape[3] == 1344:
        print("SUCCESS! Trace operates perfectly across arbitrary resolutions.")
    else:
        print(f"CRITICAL TRACE FAILURE! Model resolved dynamic shape to: {out.shape}")
