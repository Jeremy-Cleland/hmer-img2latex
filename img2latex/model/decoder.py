"""
Transformer decoder for the image-to-LaTeX model.

Also keeps the original LSTMDecoder for reference; Seq2Seq uses TransformerDecoder.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class SinusoidalPositionalEncoding(nn.Module):
    """Standard 1D sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder with tied input/output embeddings.

    Cross-attends over the encoder feature grid. Training is fully parallel
    with a causal mask; inference feeds the growing prefix each step.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 141,
        pad_token_id: int = 0,
    ):
        super().__init__()
        d_model = embedding_dim
        if d_model != hidden_dim:
            logger.warning(
                "decoder hidden_dim (%s) differs from embedding_dim (%s); using embedding_dim",
                hidden_dim,
                embedding_dim,
            )
            d_model = embedding_dim

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.scale = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        if pad_token_id is not None:
            self.embedding.weight.data[pad_token_id].zero_()
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_length + 8, dropout=dropout
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        logger.info(
            "Initialized Transformer decoder: vocab=%s d_model=%s layers=%s heads=%s",
            vocab_size,
            d_model,
            num_layers,
            nhead,
        )

    def forward(
        self,
        memory: torch.Tensor,
        tgt_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            memory: Encoder grid, (batch, src_len, d_model)
            tgt_tokens: Target token ids, (batch, tgt_len)
            memory_key_padding_mask: True = ignore encoder position
            tgt_key_padding_mask: True = ignore target pad tokens

        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        tgt = self.embedding(tgt_tokens) * self.scale
        tgt = self.pos_encoding(tgt)
        seq_len = tgt_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(decoded)


class LSTMDecoder(nn.Module):
    """
    Legacy LSTM decoder kept so existing imports continue to resolve.

    Attention now expects a real encoder sequence (batch, src_len, dim), not a
    single pooled vector.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = None,
        hidden_dim: int = None,
        max_seq_length: int = None,
        lstm_layers: int = None,
        dropout: float = None,
        attention: bool = True,
    ):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = 256
        if hidden_dim is None:
            hidden_dim = 256
        if max_seq_length is None:
            max_seq_length = 141
        if lstm_layers is None:
            lstm_layers = 1
        if dropout is None:
            dropout = 0.1

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.use_attention = attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=2 * embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        if attention:
            self.attention = Attention(hidden_dim, embedding_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, encoder_output: torch.Tensor, target_sequence: torch.Tensor, hidden=None
    ) -> torch.Tensor:
        batch_size, seq_length = target_sequence.shape
        embedded = self.embedding(target_sequence)
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)

        if not self.use_attention:
            encoder_output_repeated = encoder_output.mean(dim=1, keepdim=True).repeat(1, seq_length, 1)
            lstm_input = self.dropout_layer(torch.cat([embedded, encoder_output_repeated], dim=2))
            lstm_output, _ = self.lstm(lstm_input, hidden)
            return self.output_layer(self.dropout_layer(lstm_output))

        if hidden is None:
            h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=target_sequence.device)
            c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=target_sequence.device)
            hidden = (h_0, c_0)

        embedded = self.dropout_layer(embedded)
        outputs = []
        h, c = hidden
        for t in range(seq_length):
            current_input = embedded[:, t, :].unsqueeze(1)
            context = self.attention(h[-1].unsqueeze(1), encoder_output)
            lstm_output, (h, c) = self.lstm(torch.cat([current_input, context], dim=2), (h, c))
            outputs.append(self.output_layer(self.dropout_layer(lstm_output)))
        return torch.cat(outputs, dim=1)

    def decode_step(
        self, encoder_output: torch.Tensor, input_token: torch.Tensor, hidden=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = input_token.shape[0]
        embedded = self.embedding(input_token)
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)
        if hidden is None:
            h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=input_token.device)
            c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=input_token.device)
            hidden = (h_0, c_0)
        h, c = hidden
        if self.use_attention:
            context = self.attention(h[-1].unsqueeze(1), encoder_output)
        else:
            context = encoder_output.mean(dim=1, keepdim=True)
        lstm_output, hidden = self.lstm(torch.cat([embedded, context], dim=2), hidden)
        return self.output_layer(lstm_output), hidden


class Attention(nn.Module):
    """Additive attention over an encoder sequence."""

    def __init__(self, hidden_dim: int, encoder_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1).unsqueeze(1)
        return torch.bmm(attention_weights, encoder_outputs)
