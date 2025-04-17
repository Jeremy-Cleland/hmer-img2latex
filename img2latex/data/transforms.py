"""
Custom transforms for image preprocessing.
"""

import torchvision.transforms.functional as TF
from PIL import Image


class ResizeWithAspectRatio:
    """
    Resize image to target height while maintaining aspect ratio,
    then pad OR CROP to target width. Ensures exact output dimensions.

    Operates on PIL Images. Picklable for multiprocessing.
    """

    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
        # Determine resampling filter based on Pillow version
        try:
            self.resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            self.resample_filter = Image.LANCZOS  # Fallback for older Pillow

    def __call__(self, img):
        width, height = img.size
        if height == 0:  # Avoid division by zero
            return Image.new(img.mode, (self.target_width, self.target_height), 255)

        # Calculate width needed to preserve aspect ratio for target height
        aspect_ratio = width / height
        new_width = int(
            round(self.target_height * aspect_ratio)
        )  # Use round for possibly better results

        # Resize height first
        img_resized = img.resize((new_width, self.target_height), self.resample_filter)

        # Now ensure width is exactly target_width
        if new_width == self.target_width:
            return img_resized
        elif new_width < self.target_width:
            # Pad width (right padding with white)
            padded_img = Image.new(
                img.mode, (self.target_width, self.target_height), 255
            )  # White padding
            padded_img.paste(img_resized, (0, 0))  # Paste resized image at top-left
            return padded_img
        else:  # new_width > self.target_width
            # Crop width to target width (center crop is often better than left crop)
            left = (new_width - self.target_width) // 2
            right = left + self.target_width
            # Box is (left, upper, right, lower)
            img_cropped = img_resized.crop((left, 0, right, self.target_height))
            return img_cropped
