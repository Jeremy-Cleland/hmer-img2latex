"""
Custom transforms for image preprocessing.
"""

from typing import Tuple, Union

from PIL import Image


class ResizeWithAspectRatio:
    """
    Scale the image to fit inside (target_height, target_width), then pad.

    Never crops. Returns the PIL image plus the unpadded (width, height) so
    the encoder can build a spatial padding mask.
    """

    def __init__(self, target_height, target_width, pad_value=255):
        self.target_height = target_height
        self.target_width = target_width
        self.pad_value = pad_value
        try:
            self.resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            self.resample_filter = Image.LANCZOS

    def __call__(self, img) -> Union[Image.Image, Tuple[Image.Image, int, int]]:
        width, height = img.size
        if height == 0 or width == 0:
            blank = Image.new(img.mode, (self.target_width, self.target_height), self.pad_value)
            return blank, 1, 1

        scale = min(self.target_height / height, self.target_width / width)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        img_resized = img.resize((new_width, new_height), self.resample_filter)

        canvas = Image.new(img.mode, (self.target_width, self.target_height), self.pad_value)
        canvas.paste(img_resized, (0, 0))
        return canvas, new_width, new_height
