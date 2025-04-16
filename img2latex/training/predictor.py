"""
Prediction logic for the image-to-LaTeX model.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import load_image
from img2latex.model.seq2seq import Seq2SeqModel
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import set_device

logger = get_logger(__name__)


class Predictor:
    """
    Predictor for the image-to-LaTeX model.
    
    This class handles inference with the trained model.
    """
    
    def __init__(
        self,
        model: Seq2SeqModel,
        tokenizer: LaTeXTokenizer,
        device: Optional[torch.device] = None,
        model_type: str = "cnn_lstm"
    ):
        """
        Initialize the predictor.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer for LaTeX formulas
            device: Device to use for inference
            model_type: Type of model ("cnn_lstm" or "resnet_lstm")
        """
        # Set device
        if device is None:
            self.device = set_device()
        else:
            self.device = device
        
        # Set model and tokenizer
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_type = model_type
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Initialized predictor for model type {model_type} on device {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> "Predictor":
        """
        Create a predictor from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            device: Device to use for inference
            
        Returns:
            Initialized predictor
        """
        # Set device
        if device is None:
            device = set_device()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract configuration
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        model_type = model_config.get("name", "cnn_lstm")
        
        # Create tokenizer
        tokenizer_config = checkpoint.get("tokenizer_config", {})
        tokenizer = LaTeXTokenizer(
            special_tokens=tokenizer_config.get("special_tokens"),
            max_sequence_length=tokenizer_config.get("max_sequence_length", 150)
        )
        tokenizer.token_to_id = tokenizer_config.get("token_to_id", {})
        tokenizer.id_to_token = {
            idx: token for token, idx in tokenizer.token_to_id.items()
        }
        tokenizer.vocab_size = len(tokenizer.token_to_id)
        
        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.special_tokens["PAD"]]
        tokenizer.start_token_id = tokenizer.token_to_id[tokenizer.special_tokens["START"]]
        tokenizer.end_token_id = tokenizer.token_to_id[tokenizer.special_tokens["END"]]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.special_tokens["UNK"]]
        
        # Create model
        encoder_params = model_config.get("encoder", {})
        if model_type == "cnn_lstm":
            encoder_params = encoder_params.get("cnn", {})
        else:
            encoder_params = encoder_params.get("resnet", {})
        
        decoder_params = model_config.get("decoder", {})
        
        # Set embedding dimension
        embedding_dim = model_config.get("embedding_dim", 256)
        encoder_params["embedding_dim"] = embedding_dim
        
        # Create model
        model = Seq2SeqModel(
            model_type=model_type,
            vocab_size=tokenizer.vocab_size,
            encoder_params=encoder_params,
            decoder_params=decoder_params
        )
        
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create predictor
        predictor = cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_type=model_type
        )
        
        logger.info(f"Created predictor from checkpoint {checkpoint_path}")
        return predictor
    
    def predict(
        self,
        image: Union[str, torch.Tensor, np.ndarray, Image.Image],
        beam_size: int = 0,
        max_length: int = 150,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0
    ) -> str:
        """
        Predict LaTeX formula for an image.
        
        Args:
            image: Image to predict (path, tensor, array, or PIL image)
            beam_size: Beam size for beam search (0 for greedy search)
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Predicted LaTeX formula
        """
        # Prepare image tensor
        img_tensor = self._prepare_image(image)
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        
        # Get special token IDs
        start_token_id = self.tokenizer.start_token_id
        end_token_id = self.tokenizer.end_token_id
        
        # Generate sequence
        with torch.no_grad():
            sequence = self.model.inference(
                image=img_tensor,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=max_length,
                beam_size=beam_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Convert sequence to LaTeX
        # Skip start token if present at the beginning
        if sequence[0] == start_token_id:
            sequence = sequence[1:]
        # Remove end token if present at the end
        if sequence and sequence[-1] == end_token_id:
            sequence = sequence[:-1]
        
        # Decode sequence
        latex = self.tokenizer.decode(sequence)
        
        return latex
    
    def predict_batch(
        self,
        images: List[Union[str, torch.Tensor, np.ndarray, Image.Image]],
        beam_size: int = 0,
        max_length: int = 150,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        batch_size: int = 16
    ) -> List[str]:
        """
        Predict LaTeX formulas for a batch of images.
        
        Args:
            images: List of images to predict
            beam_size: Beam size for beam search (0 for greedy search)
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            batch_size: Batch size for prediction
            
        Returns:
            List of predicted LaTeX formulas
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Prepare image tensors
            img_tensors = [self._prepare_image(img) for img in batch_images]
            batch_tensor = torch.stack(img_tensors).to(self.device)
            
            # Get special token IDs
            start_token_id = self.tokenizer.start_token_id
            end_token_id = self.tokenizer.end_token_id
            
            # Generate sequences for the batch
            with torch.no_grad():
                for j in range(len(batch_images)):
                    img_tensor = batch_tensor[j:j+1]  # Keep batch dimension
                    
                    sequence = self.model.inference(
                        image=img_tensor,
                        start_token_id=start_token_id,
                        end_token_id=end_token_id,
                        max_length=max_length,
                        beam_size=beam_size,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    # Convert sequence to LaTeX
                    # Skip start token if present at the beginning
                    if sequence[0] == start_token_id:
                        sequence = sequence[1:]
                    # Remove end token if present at the end
                    if sequence and sequence[-1] == end_token_id:
                        sequence = sequence[:-1]
                    
                    # Decode sequence
                    latex = self.tokenizer.decode(sequence)
                    results.append(latex)
        
        return results
    
    def _prepare_image(
        self, 
        image: Union[str, torch.Tensor, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Prepare an image for prediction.
        
        Args:
            image: Image to prepare (path, tensor, array, or PIL image)
            
        Returns:
            Preprocessed image tensor
        """
        # Determine image size based on model type
        if self.model_type == "cnn_lstm":
            img_size = (50, 200)
            channels = 1
        else:  # resnet_lstm
            img_size = (224, 224)
            channels = 3
        
        # Handle different input types
        if isinstance(image, str):
            # Load image from path
            img_tensor = load_image(image, img_size, channels)
        elif isinstance(image, torch.Tensor):
            # Use tensor as is
            img_tensor = image
            
            # Check shape and convert if needed
            if img_tensor.dim() == 2:
                # Add channel dimension for grayscale
                img_tensor = img_tensor.unsqueeze(0)
            
            # Resize if needed
            if img_tensor.shape[-2:] != img_size:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0),
                    size=img_size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
            
            # Normalize if needed
            if img_tensor.min() < 0 or img_tensor.max() > 1:
                img_tensor = img_tensor / 255.0
                # Normalize to [-1, 1]
                img_tensor = img_tensor * 2.0 - 1.0
        elif isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            if image.ndim == 2:
                # Grayscale
                image = np.expand_dims(image, axis=0)
            elif image.ndim == 3 and image.shape[0] not in [1, 3]:
                # HWC to CHW
                image = np.transpose(image, (2, 0, 1))
            
            img_tensor = torch.from_numpy(image).float()
            
            # Resize if needed
            if img_tensor.shape[-2:] != img_size:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0),
                    size=img_size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
            
            # Normalize if needed
            if img_tensor.min() < 0 or img_tensor.max() > 1:
                img_tensor = img_tensor / 255.0
                # Normalize to [-1, 1]
                img_tensor = img_tensor * 2.0 - 1.0
        elif isinstance(image, Image.Image):
            # Convert PIL image to tensor
            if channels == 1 and image.mode != "L":
                image = image.convert("L")
            elif channels == 3 and image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize the image
            image = image.resize(img_size[::-1])  # PIL uses (width, height)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Add channel dimension for grayscale images
            if channels == 1:
                img_array = np.expand_dims(img_array, axis=0)
            else:
                # Rearrange from HWC to CHW
                img_array = np.transpose(img_array, (2, 0, 1))
            
            # Convert to torch tensor
            img_tensor = torch.from_numpy(img_array).float()
            
            # Normalize to [0, 1]
            img_tensor = img_tensor / 255.0
            
            # Normalize to [-1, 1]
            img_tensor = img_tensor * 2.0 - 1.0
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, torch.Tensor, numpy.ndarray, or PIL.Image.Image."
            )
        
        # Convert grayscale to RGB for ResNet if needed
        if self.model_type == "resnet_lstm" and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        
        # Add batch dimension if not present
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor