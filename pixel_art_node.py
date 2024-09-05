from sklearn.cluster import KMeans
from skimage import color
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

class PixelArtNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "num_colors": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                    "display": "slider"
                }),
                "pixel_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "save_format": (["bmp", "png", "webp"], {
                    "default": "png",
                    "display": "dropdown"
                }),
                "save_path": ("STRING", {
                    "default": "output_image.png",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Image/Effects"

    def execute(self, image, num_colors, pixel_size, save_format, save_path):
        batch_size, height, width, channels = image.shape
        
        # Downscale the image to pixelate
        downscale_size = (height // pixel_size, width // pixel_size)
        small_image = F.interpolate(image.permute(0, 3, 1, 2), size=downscale_size, mode='bilinear', align_corners=False)
        
        # Upscale the image back to original size
        pixelated_image = F.interpolate(small_image, size=(height, width), mode='nearest').permute(0, 2, 3, 1)

        # Convert the image from RGB to CIE Lab space
        pixelated_image_np = pixelated_image[0].cpu().numpy()
        lab_image = color.rgb2lab(pixelated_image_np)

        # Reshape the image for K-Means (combine all pixels into one long list)
        flat_lab_image = lab_image.reshape(-1, 3)

        # Apply K-Means++ clustering in CIE Lab color space
        kmeans = KMeans(n_clusters=num_colors, init="k-means++").fit(flat_lab_image)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Assign each pixel to the nearest cluster center (color)
        quantized_lab_image = centroids[labels].reshape(lab_image.shape)

        # Convert the quantized image back to RGB space
        quantized_rgb_image = color.lab2rgb(quantized_lab_image)

        # Convert the result to a tensor
        quantized_image_tensor = torch.tensor(quantized_rgb_image).unsqueeze(0).float()

        # Save the image using PIL
        pil_image = Image.fromarray((quantized_rgb_image * 255).astype('uint8'))
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pil_image.save(save_path, format=save_format.upper())

        return (quantized_image_tensor,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "PixelArtNode": PixelArtNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtNode": "Pixel Art Effect"
}
