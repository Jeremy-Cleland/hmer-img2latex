"""
Prediction logic for the image-to-LaTeX model.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import load_image_with_size
from img2latex.model.seq2seq import Seq2SeqModel
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import set_device

logger = get_logger(__name__, log_level="INFO")


class Predictor:
    """Inference wrapper around a trained Seq2SeqModel."""

    def __init__(
        self,
        model: Seq2SeqModel,
        tokenizer: LaTeXTokenizer,
        device: Optional[torch.device] = None,
        model_type: str = "cnn_transformer",
        img_size: Tuple[int, int] = (64, 512),
        channels: int = 1,
        length_penalty: float = 0.7,
    ):
        self.device = set_device() if device is None else device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.img_size = img_size
        self.channels = channels
        self.length_penalty = length_penalty
        self.model.eval()
        logger.info("Initialized predictor for %s on %s (%sx%s)", model_type, self.device, img_size[0], img_size[1])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[torch.device] = None) -> "Predictor":
        if device is None:
            device = set_device()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        model_type = model_config.get("name", "cnn_transformer")
        inf_cfg = config.get("inference", {})

        tokenizer_config = checkpoint.get("tokenizer_config", {})
        tokenizer = LaTeXTokenizer(
            special_tokens=tokenizer_config.get("special_tokens"),
            max_sequence_length=tokenizer_config.get("max_sequence_length", 141),
        )
        tokenizer.token_to_id = tokenizer_config.get("token_to_id", {})
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
        tokenizer.vocab_size = len(tokenizer.token_to_id)
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.special_tokens["PAD"]]
        tokenizer.start_token_id = tokenizer.token_to_id[tokenizer.special_tokens["START"]]
        tokenizer.end_token_id = tokenizer.token_to_id[tokenizer.special_tokens["END"]]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.special_tokens["UNK"]]

        encoder_params = model_config.get("encoder", {})
        if model_type.startswith("resnet"):
            encoder_params = encoder_params.get("resnet", encoder_params)
        else:
            encoder_params = encoder_params.get("cnn", encoder_params)
        embedding_dim = model_config.get("embedding_dim", 256)
        encoder_params = dict(encoder_params)
        encoder_params["embedding_dim"] = embedding_dim
        decoder_params = model_config.get("decoder", {})

        model = Seq2SeqModel(
            model_type=model_type,
            vocab_size=tokenizer.vocab_size,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        img_size = (encoder_params.get("img_height", 64), encoder_params.get("img_width", 512))
        channels = encoder_params.get("channels", 3 if model_type.startswith("resnet") else 1)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_type=model_type,
            img_size=img_size,
            channels=channels,
            length_penalty=inf_cfg.get("length_penalty", 0.7),
        )

    def predict(
        self,
        image: Union[str, torch.Tensor, np.ndarray, Image.Image],
        beam_size: int = 5,
        max_length: int = 141,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        length_penalty: Optional[float] = None,
    ) -> str:
        img_tensor, valid_w, valid_h = self._prepare_image(image)
        img_tensor = img_tensor.to(self.device)
        widths = torch.tensor([valid_w], device=self.device)
        heights = torch.tensor([valid_h], device=self.device)
        with torch.no_grad():
            sequence = self.model.inference(
                image=img_tensor,
                start_token_id=self.tokenizer.start_token_id,
                end_token_id=self.tokenizer.end_token_id,
                max_length=max_length,
                beam_size=beam_size,
                length_penalty=length_penalty if length_penalty is not None else self.length_penalty,
                valid_widths=widths,
                valid_heights=heights,
            )
        return self.tokenizer.decode(sequence)

    def predict_batch(
        self,
        images: List[Union[str, torch.Tensor, np.ndarray, Image.Image]],
        beam_size: int = 5,
        max_length: int = 141,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        batch_size: int = 16,
        length_penalty: Optional[float] = None,
        widths: Optional[torch.Tensor] = None,
        heights: Optional[torch.Tensor] = None,
    ) -> List[str]:
        results: List[str] = []
        penalty = length_penalty if length_penalty is not None else self.length_penalty
        tensors = []
        valid_ws = []
        valid_hs = []
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            batch_tensor = images.to(self.device)
            w = widths.to(self.device) if widths is not None else None
            h = heights.to(self.device) if heights is not None else None
            with torch.no_grad():
                sequences = self.model.generate(
                    batch_tensor,
                    start_token_id=self.tokenizer.start_token_id,
                    end_token_id=self.tokenizer.end_token_id,
                    max_length=max_length,
                    beam_size=max(beam_size, 1),
                    length_penalty=penalty,
                    valid_widths=w,
                    valid_heights=h,
                )
            return [self.tokenizer.decode(seq) for seq in sequences]

        for image in images:
            tensor, vw, vh = self._prepare_image(image)
            tensors.append(tensor.squeeze(0))
            valid_ws.append(vw)
            valid_hs.append(vh)
        for i in range(0, len(tensors), batch_size):
            batch_tensor = torch.stack(tensors[i : i + batch_size]).to(self.device)
            w = torch.tensor(valid_ws[i : i + batch_size], device=self.device)
            h = torch.tensor(valid_hs[i : i + batch_size], device=self.device)
            with torch.no_grad():
                sequences = self.model.generate(
                    batch_tensor,
                    start_token_id=self.tokenizer.start_token_id,
                    end_token_id=self.tokenizer.end_token_id,
                    max_length=max_length,
                    beam_size=max(beam_size, 1),
                    length_penalty=penalty,
                    valid_widths=w,
                    valid_heights=h,
                )
            results.extend(self.tokenizer.decode(seq) for seq in sequences)
        return results

    def _prepare_image(
        self, image: Union[str, torch.Tensor, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, int, int]:
        if isinstance(image, str):
            tensor, vw, vh = load_image_with_size(image, self.img_size, self.channels)
        elif isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            vw, vh = tensor.shape[-1], tensor.shape[-2]
            return tensor, vw, vh
        elif isinstance(image, Image.Image):
            from img2latex.data.transforms import ResizeWithAspectRatio
            from img2latex.data.utils import pil_to_tensor

            img = image
            if self.channels == 1 and img.mode != "L":
                img = img.convert("L")
            elif self.channels == 3 and img.mode != "RGB":
                img = img.convert("RGB")
            padded, vw, vh = ResizeWithAspectRatio(self.img_size[0], self.img_size[1])(img)
            tensor = pil_to_tensor(padded, channels=self.channels)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if self.model_type.startswith("resnet") and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor, vw, vh
