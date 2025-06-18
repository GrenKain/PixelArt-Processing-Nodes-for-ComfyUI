from .pixel_art_node import PixelArtNode  # The original node
from .pixel_art_downscale_node import PixelArtDownscaleNode  # The new downscaling node
from .pixel_art_downscale_target_size_node import PixelArtDownscaleTargetSizeNode  # The new downscaling node to target size

# Register both nodes
NODE_CLASS_MAPPINGS = {
    "PixelArtNode": PixelArtNode,
    "PixelArtDownscaleNode": PixelArtDownscaleNode,
    "PixelArtDownscaleTargetSizeNode": PixelArtDownscaleTargetSizeNode,
}

# Display names for both nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtNode": "Pixel Art Node",                # Original pixel art node
    "PixelArtDownscaleNode": "Pixel Art Downscaling", # New downscaling node
    "PixelArtDownscaleTargetSizeNode": "Pixel Art Downscaling To Target Size" # New downscaling node
}
