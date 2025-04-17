"""
LSTM-based decoder for the image-to-LaTeX model.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder for the image-to-LaTeX model.

    This decoder processes the encoder output and generates LaTeX tokens
    one at a time using an LSTM network.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        max_seq_length: int = 141,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        attention: bool = False,
    ):
        """
        Initialize the LSTM decoder.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the input and token embeddings
            hidden_dim: Dimension of the LSTM hidden state
            max_seq_length: Maximum length of generated sequences
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            attention: Whether to use attention mechanism
        """
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.use_attention = attention

        # Embedding layer for the input tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        # Input consists of: token embedding + encoder output
        # So total input size is embedding_dim + embedding_dim = 2 * embedding_dim
        self.lstm = nn.LSTM(
            input_size=2 * embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention mechanism
        if attention:
            self.attention = Attention(hidden_dim, embedding_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        logger.info(
            f"Initialized LSTM decoder with vocab size: {vocab_size}, hidden dim: {hidden_dim}"
        )

    def forward(
        self, encoder_output: torch.Tensor, target_sequence: torch.Tensor, hidden=None
    ) -> torch.Tensor:
        """
        Forward pass through the LSTM decoder (training mode).

        Args:
            encoder_output: Output from the encoder, shape (batch_size, embedding_dim)
            target_sequence: Input token sequence, shape (batch_size, seq_length)
            hidden: Initial hidden state for the LSTM

        Returns:
            Logits for each token in the output sequence, shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = target_sequence.shape

        # Get token embeddings
        embedded = self.embedding(
            target_sequence
        )  # (batch_size, seq_length, embedding_dim)

        if not self.use_attention:
            # Repeat encoder output for each time step
            # Shape: (batch_size, seq_length, embedding_dim)
            encoder_output_repeated = encoder_output.unsqueeze(1).repeat(
                1, seq_length, 1
            )

            # Concatenate token embeddings with encoder output
            # Shape: (batch_size, seq_length, 2*embedding_dim)
            lstm_input = torch.cat([embedded, encoder_output_repeated], dim=2)

            # Apply dropout to the input
            lstm_input = self.dropout_layer(lstm_input)

            # Pass through LSTM
            lstm_output, hidden = self.lstm(lstm_input, hidden)

            # Apply dropout to LSTM output
            lstm_output = self.dropout_layer(lstm_output)

            # Project to vocabulary size
            # Shape: (batch_size, seq_length, vocab_size)
            output = self.output_layer(lstm_output)
        else:
            # Initialize hidden state if not provided
            if hidden is None:
                h_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=target_sequence.device,
                )
                c_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=target_sequence.device,
                )
                hidden = (h_0, c_0)

            # Apply dropout to the embedded tokens
            embedded = self.dropout_layer(embedded)

            # Process one time step at a time to apply attention
            outputs = []
            h, c = hidden

            for t in range(seq_length):
                # Get the current token embedding
                current_input = embedded[:, t, :].unsqueeze(
                    1
                )  # (batch_size, 1, embedding_dim)

                # Apply attention
                context = self.attention(
                    h[-1].unsqueeze(1), encoder_output.unsqueeze(1)
                )

                # Concatenate with current token embedding
                lstm_input = torch.cat([current_input, context], dim=2)

                # Pass through LSTM
                lstm_output, (h, c) = self.lstm(lstm_input, (h, c))

                # Apply dropout
                lstm_output = self.dropout_layer(lstm_output)

                # Project to vocabulary size
                output_t = self.output_layer(lstm_output)
                outputs.append(output_t)

            # Stack outputs
            output = torch.cat(outputs, dim=1)  # (batch_size, seq_length, vocab_size)

        return output

    def decode_step(
        self, encoder_output: torch.Tensor, input_token: torch.Tensor, hidden=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoding step for inference.

        Args:
            encoder_output: Output from the encoder, shape (batch_size, embedding_dim)
            input_token: Input token tensor, shape (batch_size, 1)
            hidden: Hidden state from previous step

        Returns:
            Tuple of (output_logits, new_hidden_state)
        """
        batch_size = input_token.shape[0]

        # Get token embedding
        embedded = self.embedding(input_token)  # (batch_size, 1, embedding_dim)

        if not self.use_attention:
            # No Attention
            encoder_output_repeated = encoder_output.unsqueeze(1)

            # --- Sanity Check Dimensions Before Concatenation ---
            if embedded.ndim != 3 or encoder_output_repeated.ndim != 3:
                raise RuntimeError(
                    f"Shape mismatch before cat! embedded: {embedded.shape}, "
                    f"encoder_output_repeated: {encoder_output_repeated.shape}"
                )
            # ----------------------------------------------------

            lstm_input = torch.cat([embedded, encoder_output_repeated], dim=2)

            # Initialize hidden state if not provided
            if hidden is None:
                h_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                c_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                hidden = (h_0, c_0)

            # Pass through LSTM
            lstm_output, hidden = self.lstm(lstm_input, hidden)

            # Project to vocabulary size
            output = self.output_layer(lstm_output)
        else:
            # Initialize hidden state if not provided
            if hidden is None:
                h_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                c_0 = torch.zeros(
                    self.lstm_layers,
                    batch_size,
                    self.hidden_dim,
                    device=input_token.device,
                )
                hidden = (h_0, c_0)

            h, c = hidden

            # Apply attention
            context = self.attention(h[-1].unsqueeze(1), encoder_output.unsqueeze(1))

            # Concatenate with current token embedding
            lstm_input = torch.cat([embedded, context], dim=2)

            # Pass through LSTM
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))

            # Project to vocabulary size
            output = self.output_layer(lstm_output)

            hidden = (h, c)

        return output, hidden


class Attention(nn.Module):
    """
    Attention mechanism for the decoder.

    This attention layer allows the decoder to focus on different parts
    of the encoder output at each decoding step.
    """

    def __init__(self, hidden_dim: int, encoder_dim: int):
        """
        Initialize the attention layer.

        Args:
            hidden_dim: Dimension of the decoder hidden state
            encoder_dim: Dimension of the encoder output
        """
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        # Attention layers
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.

        Args:
            hidden: Decoder hidden state, shape (batch_size, 1, hidden_dim)
            encoder_outputs: Encoder outputs, shape (batch_size, seq_length, encoder_dim)

        Returns:
            Context vector, shape (batch_size, 1, encoder_dim)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state
        hidden = hidden.repeat(1, src_len, 1)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1).unsqueeze(1)

        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)

        return context
