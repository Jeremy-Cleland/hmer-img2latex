"""
Prediction logic for the image-to-LaTeX model.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
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
        model_type: str = "cnn_lstm",
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

        logger.info(
            f"Initialized predictor for model type {model_type} on device {self.device}"
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: Optional[torch.device] = None
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
            max_sequence_length=tokenizer_config.get("max_sequence_length", 141),
        )
        tokenizer.token_to_id = tokenizer_config.get("token_to_id", {})
        tokenizer.id_to_token = {
            idx: token for token, idx in tokenizer.token_to_id.items()
        }
        tokenizer.vocab_size = len(tokenizer.token_to_id)

        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.special_tokens["PAD"]]
        tokenizer.start_token_id = tokenizer.token_to_id[
            tokenizer.special_tokens["START"]
        ]
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
            decoder_params=decoder_params,
        )

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create predictor
        predictor = cls(
            model=model, tokenizer=tokenizer, device=device, model_type=model_type
        )

        logger.info(f"Created predictor from checkpoint {checkpoint_path}")
        return predictor

    def predict(
        self,
        image: Union[str, torch.Tensor, np.ndarray, Image.Image],
        beam_size: int = 0,
        max_length: int = 141,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
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
                top_p=top_p,
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
        max_length: int = 141,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        batch_size: int = 16,
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
            batch_images = images[i : i + batch_size]

            # Prepare image tensors
            img_tensors = [self._prepare_image(img) for img in batch_images]
            batch_tensor = torch.stack(img_tensors).to(self.device)

            # Get special token IDs
            start_token_id = self.tokenizer.start_token_id
            end_token_id = self.tokenizer.end_token_id

            # For greedy search (non-beam search), we can process the whole batch at once
            if beam_size == 0:
                # Generate sequences for the batch efficiently with greedy search
                with torch.no_grad():
                    # Process the entire batch at once
                    encoder_outputs = self.model.encoder(batch_tensor)

                    # Initialize batch sequences with start tokens
                    batch_size = batch_tensor.size(0)
                    device = batch_tensor.device
                    sequences = [[start_token_id] for _ in range(batch_size)]

                    # Initialize hidden states
                    hidden = None

                    # Generate tokens one by one for all sequences in the batch
                    for _ in range(max_length):
                        # Create a tensor with current tokens from all sequences
                        current_tokens = [seq[-1] for seq in sequences]
                        input_tokens = torch.tensor(
                            [[token] for token in current_tokens], device=device
                        )

                        # Reshape encoder outputs for the step
                        step_encoder_outputs = encoder_outputs

                        # Decode next tokens
                        with torch.no_grad():
                            # Get next token probabilities
                            outputs = []
                            for j, token in enumerate(input_tokens):
                                # Get a slice of encoder output for this sequence
                                encoder_output = step_encoder_outputs[j : j + 1]

                                output, new_hidden = self.model.decoder.decode_step(
                                    encoder_output=encoder_output,
                                    input_token=token,
                                    hidden=hidden,
                                )
                                outputs.append(output)

                            # Get the predicted tokens
                            if temperature > 0 and (top_k > 0 or top_p > 0):
                                # Apply temperature and top-k/top-p sampling
                                next_tokens = []
                                for output in outputs:
                                    logits = output.squeeze(1) / temperature
                                    probs = torch.softmax(logits, dim=-1)

                                    # Apply top-k sampling
                                    if top_k > 0:
                                        top_k_probs, top_k_indices = torch.topk(
                                            probs, top_k
                                        )
                                        probs_dist = torch.zeros_like(probs)
                                        probs_dist.scatter_(
                                            -1, top_k_indices, top_k_probs
                                        )
                                        probs = probs_dist / probs_dist.sum()

                                    # Apply top-p sampling
                                    if top_p > 0:
                                        sorted_probs, sorted_indices = torch.sort(
                                            probs, descending=True
                                        )
                                        cumulative_probs = torch.cumsum(
                                            sorted_probs, dim=-1
                                        )

                                        # Remove tokens with cumulative probability above threshold
                                        sorted_indices_to_remove = (
                                            cumulative_probs > top_p
                                        )

                                        # Shift indices to the right to keep first token above threshold
                                        sorted_indices_to_remove[..., 1:] = (
                                            sorted_indices_to_remove[..., :-1].clone()
                                        )
                                        sorted_indices_to_remove[..., 0] = 0

                                        indices_to_remove = (
                                            sorted_indices_to_remove.scatter(
                                                -1,
                                                sorted_indices,
                                                sorted_indices_to_remove,
                                            )
                                        )
                                        probs[indices_to_remove] = 0
                                        probs = probs / probs.sum()

                                    # Sample from the distribution
                                    next_token = torch.multinomial(probs, 1).item()
                                    next_tokens.append(next_token)
                            else:
                                # Greedy search - take argmax
                                next_tokens = [
                                    torch.argmax(output.squeeze(1), dim=-1).item()
                                    for output in outputs
                                ]

                        # Add tokens to sequences
                        for j, token in enumerate(next_tokens):
                            sequences[j].append(token)

                        # Check if all sequences have reached end token
                        if all(seq[-1] == end_token_id for seq in sequences):
                            break
            else:
                # For beam search, process each image individually
                with torch.no_grad():
                    sequences = []
                    for j in range(batch_tensor.size(0)):
                        img_tensor = batch_tensor[j : j + 1]  # Keep batch dimension

                        sequence = self.model.inference(
                            image=img_tensor,
                            start_token_id=start_token_id,
                            end_token_id=end_token_id,
                            max_length=max_length,
                            beam_size=beam_size,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                        sequences.append(sequence)

            # Convert sequences to LaTeX
            for sequence in sequences:
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
        self, image: Union[str, torch.Tensor, np.ndarray, Image.Image]
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
            img_size = (64, 800)
            channels = 1
        else:  # resnet_lstm
            img_size = (64, 800)
            channels = 3

        # Handle different input types
        if isinstance(image, str):
            # Load image from path using the utility function
            img_tensor = load_image(image, img_size, channels)
        elif isinstance(image, torch.Tensor):
            # For tensor input, process it appropriately
            img_tensor = self._preprocess_tensor(image, img_size, channels)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to tensor and then process
            tensor_image = self._numpy_to_tensor(image, channels)
            img_tensor = self._preprocess_tensor(tensor_image, img_size, channels)
        elif isinstance(image, Image.Image):
            # Convert PIL image to tensor using the utility function logic
            # First convert and resize
            if channels == 1 and image.mode != "L":
                image = image.convert("L")
            elif channels == 3 and image.mode != "RGB":
                image = image.convert("RGB")

            # Resize the image
            image = image.resize(img_size[::-1])  # PIL uses (width, height)

            # Convert to tensor using NumPy as intermediate
            img_array = np.array(image)
            if channels == 1:
                img_array = np.expand_dims(img_array, axis=0)
            else:
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW

            img_tensor = torch.from_numpy(img_array).float() / 255.0
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

    def _preprocess_tensor(
        self, tensor: torch.Tensor, img_size: Tuple[int, int], channels: int
    ) -> torch.Tensor:
        """
        Preprocess a tensor image.

        Args:
            tensor: Image tensor
            img_size: Target image size (height, width)
            channels: Number of channels (1 for grayscale, 3 for RGB)

        Returns:
            Preprocessed tensor
        """
        # Add channel dimension for grayscale if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Resize if needed
        if tensor.shape[-2:] != img_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0) if tensor.dim() == 3 else tensor,
                size=img_size,
                mode="bilinear",
                align_corners=False,
            )
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

        # Normalize if needed
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = tensor / 255.0
            # Normalize to [-1, 1]
            tensor = tensor * 2.0 - 1.0

        return tensor

    def _numpy_to_tensor(self, array: np.ndarray, channels: int) -> torch.Tensor:
        """
        Convert a numpy array to a tensor with proper channel format.

        Args:
            array: Numpy array
            channels: Number of channels (1 for grayscale, 3 for RGB)

        Returns:
            Tensor with proper channel format
        """
        # Handle different array formats
        if array.ndim == 2:
            # Grayscale image without channel dimension
            array = np.expand_dims(array, axis=0)
        elif array.ndim == 3 and array.shape[0] not in [1, 3]:
            # HWC format, convert to CHW
            array = np.transpose(array, (2, 0, 1))

        # Convert to torch tensor
        return torch.from_numpy(array).float()
