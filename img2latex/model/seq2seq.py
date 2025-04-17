"""
Sequence-to-sequence model for image-to-LaTeX conversion.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from img2latex.model.decoder import LSTMDecoder
from img2latex.model.encoder import CNNEncoder, ResNetEncoder
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-sequence model for image-to-LaTeX conversion.

    This model consists of an encoder (CNN or ResNet) that processes the input image
    and a decoder (LSTM) that generates the output LaTeX sequence.
    """

    def __init__(
        self,
        model_type: str = "cnn_lstm",
        vocab_size: int = None,
        encoder_params: Dict = None,
        decoder_params: Dict = None,
    ):
        """
        Initialize the sequence-to-sequence model.

        Args:
            model_type: Type of model to use, either "cnn_lstm" or "resnet_lstm"
            vocab_size: Size of the vocabulary
            encoder_params: Parameters for the encoder
            decoder_params: Parameters for the decoder
        """
        super(Seq2SeqModel, self).__init__()

        # Set default parameters if not provided
        if encoder_params is None:
            encoder_params = {}
        if decoder_params is None:
            decoder_params = {}

        # Set default vocab_size if not provided
        if vocab_size is None:
            vocab_size = 100  # Fallback only if no value provided

        # Get embedding dimension from encoder params
        embedding_dim = encoder_params.get("embedding_dim", 256)

        # Initialize encoder
        if model_type == "cnn_lstm":
            self.encoder = CNNEncoder(
                img_height=encoder_params.get("img_height", 50),
                img_width=encoder_params.get("img_width", 200),
                channels=encoder_params.get("channels", 1),
                conv_filters=encoder_params.get("conv_filters", [32, 64, 128]),
                kernel_size=encoder_params.get("kernel_size", 3),
                pool_size=encoder_params.get("pool_size", 2),
                padding=encoder_params.get("padding", "same"),
                embedding_dim=embedding_dim,
            )
        elif model_type == "resnet_lstm":
            self.encoder = ResNetEncoder(
                img_height=encoder_params.get("img_height", 224),
                img_width=encoder_params.get("img_width", 224),
                channels=encoder_params.get("channels", 3),
                model_name=encoder_params.get("model_name", "resnet50"),
                embedding_dim=embedding_dim,
                freeze_backbone=encoder_params.get("freeze_backbone", True),
            )
        else:
            raise ValueError(
                f"Invalid model type: {model_type}. Expected 'cnn_lstm' or 'resnet_lstm'."
            )

        # Initialize decoder
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=decoder_params.get("hidden_dim", 256),
            max_seq_length=decoder_params.get("max_seq_length", 150),
            lstm_layers=decoder_params.get("lstm_layers", 1),
            dropout=decoder_params.get("dropout", 0.1),
            attention=decoder_params.get("attention", False),
        )

        self.model_type = model_type
        self.vocab_size = vocab_size

        logger.info(f"Initialized {model_type} model with vocab size: {vocab_size}")

    def forward(
        self, images: torch.Tensor, target_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the sequence-to-sequence model (training mode).

        Args:
            images: Input images, shape (batch_size, channels, height, width)
            target_sequences: Target LaTeX sequences, shape (batch_size, seq_length)

        Returns:
            Output logits, shape (batch_size, seq_length, vocab_size)
        """
        # Encode the images
        encoder_output = self.encoder(images)

        # Decode the sequences
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            target_sequence=target_sequences[
                :, :-1
            ],  # Exclude the last token (end token)
        )

        return decoder_output

    def inference(
        self,
        image: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        beam_size: int = None,
    ) -> List[int]:
        """
        Generate a LaTeX sequence for an input image (inference mode).

        Args:
            image: Input image, shape (batch_size, channels, height, width)
            start_token_id: ID of the start token
            end_token_id: ID of the end token
            max_length: Maximum length of the generated sequence
            temperature: Softmax temperature (higher values produce more diverse outputs)
            top_k: If > 0, only sample from the top k most probable tokens
            top_p: If > 0.0, only sample from the top tokens with cumulative probability >= top_p
            beam_size: If > 0, use beam search with the specified beam size

        Returns:
            List of token IDs for the generated sequence
        """
        # Set default values if not provided
        if max_length is None:
            max_length = 150  # Fallback only if config value not passed
        if temperature is None:
            temperature = 1.0  # Fallback only if config value not passed
        if top_k is None:
            top_k = 0  # Fallback only if config value not passed
        if top_p is None:
            top_p = 0.0  # Fallback only if config value not passed
        if beam_size is None:
            beam_size = 0  # Fallback only if config value not passed

        # Encode the image
        encoder_output = self.encoder(image)

        # Handle batch size 1 case for inference
        if encoder_output.dim() == 1:
            encoder_output = encoder_output.unsqueeze(0)

        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        if beam_size > 0:
            return self._beam_search(
                encoder_output=encoder_output,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=max_length,
                beam_size=beam_size,
            )
        else:
            return self._greedy_search(
                encoder_output=encoder_output,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
    
    def _greedy_search(
        self,
        encoder_output: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[int]:
        """Greedy decoding: pick highest-probability token at each step."""
        device = encoder_output.device
        batch_size = encoder_output.size(0)
        # Initialize input with start tokens
        input_tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        hidden = None
        sequences = [[start_token_id] for _ in range(batch_size)]

        for _ in range(max_length):
            output, hidden = self.decoder.decode_step(encoder_output, input_tokens, hidden)
            logits = output.squeeze(1)  # (batch_size, vocab_size)
            if temperature != 1.0:
                logits = logits / temperature
            next_tokens = torch.argmax(logits, dim=-1)  # (batch_size,)
            input_tokens = next_tokens.unsqueeze(1)
            for i in range(batch_size):
                sequences[i].append(next_tokens[i].item())
            # Stop if all sequences have generated end token
            if all(tok == end_token_id for tok in next_tokens.tolist()):
                break

        # Post-process single example
        seq = sequences[0] if batch_size == 1 else sequences
        if batch_size == 1:
            # Remove start token
            if seq and seq[0] == start_token_id:
                seq = seq[1:]
            # Truncate at end token
            if end_token_id in seq:
                seq = seq[: seq.index(end_token_id)]
        return seq

    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        beam_size: int,
    ) -> List[int]:
        """Simple beam search decoding (only supports batch_size=1)."""
        device = encoder_output.device
        if encoder_output.size(0) != 1:
            return self._greedy_search(
                encoder_output, start_token_id, end_token_id, max_length, 1.0, 0, 0.0
            )

        # Beam entries: list of dicts {'tokens': [...], 'hidden': hidden_state, 'score': float}
        beams = [{"tokens": [start_token_id], "hidden": None, "score": 0.0}]
        completed = []

        for _ in range(max_length):
            candidates = []
            for beam in beams:
                tokens = beam["tokens"]
                last_token = tokens[-1]
                if last_token == end_token_id:
                    completed.append(beam)
                    continue
                input_token = torch.tensor([[last_token]], dtype=torch.long, device=device)
                output, new_hidden = self.decoder.decode_step(
                    encoder_output, input_token, beam["hidden"]
                )
                logits = output.squeeze(1)  # (1, vocab_size)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                topk_probs, topk_idx = torch.topk(log_probs, beam_size)
                for log_p, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
                    candidates.append(
                        {
                            "tokens": tokens + [idx],
                            "hidden": (new_hidden[0].clone(), new_hidden[1].clone()),
                            "score": beam["score"] + log_p,
                        }
                    )
            if not candidates:
                break
            # Keep top beams
            candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)
            beams = candidates[:beam_size]
            # Early exit if all beams ended
            if all(b["tokens"][-1] == end_token_id for b in beams):
                completed.extend(beams)
                break

        best_beam = None
        if completed:
            best_beam = max(completed, key=lambda b: b["score"])
        else:
            best_beam = beams[0]
        seq = best_beam["tokens"]
        # Remove start token
        if seq and seq[0] == start_token_id:
            seq = seq[1:]
        # Truncate at end token
        if end_token_id in seq:
            seq = seq[: seq.index(end_token_id)]
        return seq
