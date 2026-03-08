import os
import sys
import gc
import argparse
import glob
import re

import numpy as np
import torch
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

# Add NormalCrafter to sys.path so we can import its modules
TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMALCRAFTER_DIR = os.path.join(TOOL_ROOT, "repo")
sys.path.insert(0, NORMALCRAFTER_DIR)

from diffusers.training_utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder
from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter

def load_image_sequence(input_pattern, max_res):
    """
    Loads an EXR or PNG sequence based on a printf-style pattern (e.g., input_%04d.exr).
    Returns a list of PIL Images.
    """
    # Replace printf-style pattern with glob
    if "%0" in input_pattern:
        pattern_regex = re.compile(r'%0(\d+)d')
        glob_pattern = pattern_regex.sub(r'*', input_pattern)
    else:
        glob_pattern = input_pattern
        
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise ValueError(f"No files found for pattern: {input_pattern}")

    print(f"Found {len(files)} frames.")
    frames = []
    
    # Read the first frame to get dimensions
    first_img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise ValueError(f"Could not read image: {files[0]}")
    
    original_height, original_width = first_img.shape[:2]
    
    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    else:
        height = original_height
        width = original_width

    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        
        # Convert EXR (BGR/BGRA float) -> RGB uint8 for NormalCrafter (it expects 0-255 images)
        if img.dtype == np.float32 or img.dtype == np.float16:
            # Simple tonemapping/clipping for EXR if they are linear
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA@RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        frames.append(Image.fromarray(img))
    
    return frames, files

def save_exr_sequence(normals_array, output_pattern, original_files):
    """
    Saves the generated normals (numpy array of shape [N, H, W, 3]) 
    to an EXR sequence using the output pattern.
    """
    # NormalCrafter models output range could be arbitrary, but `vis_sequence_normal` 
    # clips to [-1, 1]. In Nuke, vector space [-1, 1] is standard for Normal maps.
    normals_array = normals_array.clip(-1.0, 1.0).astype(np.float32)

    output_dir = os.path.dirname(output_pattern)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, normal_frame in enumerate(normals_array):
        # Determine the frame number from the original file 
        # (assuming simple sequential index if parsing is too complex, but let's just use 1-based index or parse it)
        try:
            # Extract number from original file
            frame_num_str = re.findall(r'\d+', os.path.basename(original_files[i]))[-1]
            frame_num = int(frame_num_str)
        except:
            frame_num = i + 1

        if "%0" in output_pattern:
            # Nuke style pattern %04d
            ext_idx = output_pattern.rfind('.')
            base = output_pattern[:ext_idx]
            ext = output_pattern[ext_idx:]
            # Replace %04d with correctly padded number
            match = re.search(r'%0(\d+)d', base)
            if match:
                pad = int(match.group(1))
                out_path = re.sub(r'%0\d+d', f"{frame_num:0{pad}d}", output_pattern)
            else:
                out_path = f"{base}_{frame_num:04d}{ext}"
        else:
            # Just append sequence
            out_path = f"{output_pattern}_{frame_num:04d}.exr"
            
        # OpenCV expects BGR
        normal_bgr = normal_frame[:, :, ::-1]
        cv2.imwrite(out_path, normal_bgr)
        print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="NormalCrafter Nuke Execution Wrapper")
    parser.add_argument("--input", type=str, required=True, help="Input sequence pattern (e.g. input_%04d.exr)")
    parser.add_argument("--output", type=str, required=True, help="Output sequence pattern (e.g. output_%04d.exr)")
    parser.add_argument("--unet-path", type=str, default="Yanrui95/NormalCrafter")
    parser.add_argument("--pre-train-path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--max-res", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--decode-chunk-size", type=int, default=7)
    parser.add_argument("--time-step-size", type=int, default=10)
    parser.add_argument("--cpu-offload", type=str, default="model")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # Load input frames
    frames, original_files = load_image_sequence(args.input, args.max_res)

    print("Loading Models...")
    weight_dtype = torch.float16
    
    # Load UNet
    unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
        args.unet_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    
    # Load VAE
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.unet_path, subfolder="vae"
    )
    
    vae.to(dtype=weight_dtype)
    unet.to(dtype=weight_dtype)
    
    pipe = NormalCrafterPipeline.from_pretrained(
        args.pre_train_path,
        unet=unet,
        vae=vae,
        torch_dtype=weight_dtype,
        variant="fp16",
    )

    if args.cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif args.cpu_offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention.")
    except Exception as e:
        print("Xformers not enabled:", e)

    set_seed(args.seed)

    print(f"Starting inference on {len(frames)} frames...")
    with torch.inference_mode():
        res = pipe(
            frames,
            decode_chunk_size=args.decode_chunk_size,
            time_step_size=args.time_step_size,
            window_size=args.window_size,
        ).frames[0]
        
    print("Inference complete. Saving EXR sequence...")
    # res is typically a list of np.arrays or a batched numpy array [N, H, W, 3]
    if isinstance(res, list):
        res = np.stack(res)
    
    save_exr_sequence(res, args.output, original_files)

    print("Done!")

if __name__ == "__main__":
    main()
