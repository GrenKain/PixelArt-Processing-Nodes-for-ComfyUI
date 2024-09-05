# PixelArt Processing Nodes for ComfyUI

This repository provides two custom nodes for ComfyUI that allow for pixel art image processing, including quantization, upscaling, and downscaling while preserving pixelated details.

## Nodes:

### 1. Pixel Art Quantization Node
- **Description**: Converts any image into pixel art by reducing resolution and merging colors into a limited palette.
- **Outputs**: 
  - A pixelated version of the input image.
  - A color palette with the selected number of colors.

### 2. Pixel Art Scaling Node
- **Description**: Scales pixel art images up or down while maintaining crisp edges. Supports nearest-neighbor and bicubic scaling methods.
- **Outputs**: 
  - A scaled version of the input image.

## Usage

1. Clone or download this repository.
2. Copy the folder `pixelart_processing` into the `custom_nodes` directory of your ComfyUI installation.
3. Restart ComfyUI.
4. Find the new nodes under the **Image/Processing** category in ComfyUI.

## Features
- GPU-accelerated pixel art image processing.
- Configurable color quantization.
- Sharp edge preservation during scaling.
- Output of color palettes for further processing.

## License
MIT License. See `LICENSE` for more details.
