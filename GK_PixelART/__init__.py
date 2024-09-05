from .pixel_art_node import PixelArtNode  # The original node
from .pixel_art_downscale_node import PixelArtDownscaleNode  # The new downscaling node

# Register both nodes
NODE_CLASS_MAPPINGS = {
    "PixelArtNode": PixelArtNode,
    "PixelArtDownscaleNode": PixelArtDownscaleNode,
}

# Display names for both nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtNode": "Pixel Art Node",                # Original pixel art node
    "PixelArtDownscaleNode": "Pixel Art Downscaling" # New downscaling node
}
