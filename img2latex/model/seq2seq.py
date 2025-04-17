"""
Sequence-to-sequence model for image-to-LaTeX conversion.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from img2latex.model.decoder import LSTMDecoder
from img2latex.model.encoder import CNNEncoder, ResNetEncoder
from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-sequence model for image-to-LaTeX conversion.

    This model consists of an encoder (CNN or ResNet) that processes the input image
    and a decoder (LSTM) that generates the output LaTeX sequence.
    """

    def __init__(
        self,
        model_type: str = "cnn_lstm",
        vocab_size: int = 100,
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
        max_length: int = 150,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        beam_size: int = 0,
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

    # --- This method seems unused / incorrect for batch processing ---
    # --- Reverting to a simple single-step call ---
    # def _greedy_search(...): ...

    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        beam_size: int,
    ) -> List[int]:
        """
        Perform beam search decoding.

        Args:
            encoder_output: Output from the encoder
            start_token_id: ID of the start token
            end_token_id: ID of the end token
            max_length: Maximum length of the generated sequence
            beam_size: Beam size

        Returns:
            List of token IDs for the best sequence found
        """
        device = encoder_output.device
        batch_size = encoder_output.shape[0]

        # Repeat encoder output for each beam
        encoder_output = encoder_output.repeat_interleave(beam_size, dim=0)

        # Start with the start token
        input_token = torch.tensor([[start_token_id]], device=device)

        # Initialize sequences, scores, and finished flags
        sequences = [[start_token_id]]
        sequence_scores = torch.zeros(1, device=device)
        finished_sequences = []
        finished_scores = []

        # Initialize the hidden state
        hidden = None

        # Generate tokens one by one
        for step in range(max_length):
            # Get the number of active sequences
            num_active = len(sequences)

            # Prepare the input for the decoder
            input_tokens = torch.tensor([[seq[-1]] for seq in sequences], device=device)

            # Get the next token probabilities
            output, hidden = self.decoder.decode_step(
                encoder_output[:num_active], input_tokens, hidden
            )
            logits = output.squeeze(1)

            # Convert to log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Add the log probabilities to the sequence scores
            if step == 0:
                # For the first step, we only have one sequence
                scores = log_probs[0]
            else:
                # For subsequent steps, we have multiple sequences
                scores = sequence_scores.unsqueeze(1) + log_probs
                scores = scores.view(-1)

            # Select the top-k sequences
            if step == 0:
                top_k_scores, top_k_tokens = torch.topk(scores, beam_size)
                beam_indices = torch.zeros(beam_size, device=device, dtype=torch.long)
            else:
                top_k_scores, top_k_indices = torch.topk(
                    scores, beam_size - len(finished_sequences)
                )
                top_k_tokens = top_k_indices % self.vocab_size
                beam_indices = top_k_indices // self.vocab_size

            # Create new sequences
            new_sequences = []
            new_scores = []
            new_hidden = (
                (hidden[0][:, beam_indices, :], hidden[1][:, beam_indices, :])
                if hidden is not None
                else None
            )

            for i, (token, beam_idx) in enumerate(zip(top_k_tokens, beam_indices)):
                token_item = token.item()
                score = top_k_scores[i].item()

                # Get the sequence corresponding to the beam index
                sequence = sequences[beam_idx]

                # Check if this sequence has finished
                if token_item == end_token_id:
                    finished_sequences.append(sequence + [token_item])
                    finished_scores.append(score)
                else:
                    new_sequences.append(sequence + [token_item])
                    new_scores.append(score)

            # Break if all sequences have finished or we've reached the maximum length
            if len(finished_sequences) >= beam_size or len(new_sequences) == 0:
                break

            # Update sequences, scores, and hidden state
            sequences = new_sequences
            sequence_scores = torch.tensor(new_scores, device=device)
            hidden = new_hidden

        # If we don't have any finished sequences, use the active ones
        if len(finished_sequences) == 0:
            finished_sequences = sequences
            finished_scores = sequence_scores.tolist()

        # Sort the finished sequences by score (higher is better)
        sorted_sequences = [
            seq
            for _, seq in sorted(
                zip(finished_scores, finished_sequences),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        # Return the best sequence
        return sorted_sequences[0]
