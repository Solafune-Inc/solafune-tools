import os
import glob
import re
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import Optional

def _normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array to 0-255 uint8 range for visualization.
    Handles 16-bit Sentinel-2 data by scaling min-max.
    """
    # Handle NaN values if any
    array = np.nan_to_num(array)
    
    min_val = np.percentile(array, 2) # Percentile clipping for better contrast
    max_val = np.percentile(array, 98)
    
    if max_val == min_val:
        return np.zeros_like(array, dtype=np.uint8)
    
    # Clip and scale
    norm = np.clip((array - min_val) / (max_val - min_val), 0, 1)
    return (norm * 255).astype(np.uint8)

def _extract_date(filename: str) -> str:
    """
    Attempt to extract a date string from the filename.
    Matches YYYYMMDD, YYYY-MM-DD, or similar patterns common in satellite data.
    """
    # Common Sentinel-2 format often contains YYYYMMDD in the string
    # e.g., T54TVL_20230523T013721_B04.tif
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        
    # Standard YYYY-MM-DD
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    
    return "Unknown Date"

def create_timelapse(
    input_dir: str,
    output_filename: str,
    fps: int = 5,
    add_timestamp: bool = True,
    text_color: str = "white",
    resize_factor: float = 1.0
) -> str:
    """
    Generates an animated GIF from a sequence of GeoTIFF images in a directory.
    Useful for visualizing changes over time from satellite imagery bands.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing .tif files.
    output_filename : str
        Path where the output .gif will be saved.
    fps : int, optional
        Frames per second for the animation. Default is 5.
    add_timestamp : bool, optional
        If True, attempts to extract date from filename and overlay it. Default is True.
    text_color : str, optional
        Color of the timestamp text. Default is "white".
    resize_factor : float, optional
        Factor to resize images (e.g., 0.5 for half size). Default is 1.0 (original size).

    Returns
    -------
    str
        Path to the generated GIF file.
    """
    # Find all tif files
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")

    frames = []
    
    print(f"Processing {len(tif_files)} images for timelapse...")

    for file_path in tif_files:
        try:
            with rasterio.open(file_path) as src:
                # Read the first band (assuming single band imagery for now)
                data = src.read(1)
                
                # Normalize and convert to image
                norm_data = _normalize_array(data)
                img = Image.fromarray(norm_data, mode='L').convert("RGB")
                
                # Resize if needed
                if resize_factor != 1.0:
                    new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Add Timestamp
                if add_timestamp:
                    date_str = _extract_date(os.path.basename(file_path))
                    draw = ImageDraw.Draw(img)
                    # Use default font since we can't guarantee system fonts
                    # For a "Serious" tool we might want to carry a font, but default is safe
                    try:
                         # Try to load a larger font if possible, else default
                        font = ImageFont.truetype("arial.ttf", 20)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    # Draw text with a slight shadow/outline for visibility
                    text_position = (10, 10)
                    # Shadow
                    draw.text((text_position[0]+1, text_position[1]+1), date_str, font=font, fill="black")
                    # Main Text
                    draw.text(text_position, date_str, font=font, fill=text_color)
                
                frames.append(img)
        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)} due to error: {e}")
            continue

    if not frames:
        raise RuntimeError("No valid frames could be generated.")

    # Save as GIF
    # duration is in milliseconds per frame (1000ms / fps)
    duration = 1000 // fps
    
    frame_one = frames[0]
    frame_one.save(
        output_filename,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"Timelapse saved to {output_filename}")
    return output_filename
