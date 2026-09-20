"""
Sequence-to-sequence model: spatial encoder + Transformer decoder.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from img2latex.model.decoder import TransformerDecoder
from img2latex.model.encoder import CNNEncoder, ResNetEncoder
from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class Seq2SeqModel(nn.Module):
    """Image encoder plus Transformer decoder for LaTeX generation."""

    def __init__(
        self,
        model_type: str = "cnn_transformer",
        vocab_size: int = None,
        encoder_params: Dict = None,
        decoder_params: Dict = None,
        pad_token_id: int = 0,
    ):
        super().__init__()
        encoder_params = encoder_params or {}
        decoder_params = decoder_params or {}
        if vocab_size is None:
            vocab_size = 100

        embedding_dim = encoder_params.get("embedding_dim", 256)
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        if model_type.startswith("resnet"):
            self.encoder = ResNetEncoder(
                img_height=encoder_params.get("img_height", 64),
                img_width=encoder_params.get("img_width", 512),
                channels=encoder_params.get("channels", 3),
                model_name=encoder_params.get("model_name", "resnet18"),
                embedding_dim=embedding_dim,
                freeze_backbone=encoder_params.get("freeze_backbone", False),
            )
        else:
            self.encoder = CNNEncoder(
                img_height=encoder_params.get("img_height", 64),
                img_width=encoder_params.get("img_width", 512),
                channels=encoder_params.get("channels", 1),
                conv_filters=encoder_params.get("conv_filters", [32, 64, 128]),
                kernel_size=encoder_params.get("kernel_size", 3),
                pool_size=encoder_params.get("pool_size", 2),
                padding=encoder_params.get("padding", "same"),
                embedding_dim=embedding_dim,
            )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=decoder_params.get("hidden_dim", embedding_dim),
            nhead=decoder_params.get("nhead", 8),
            num_layers=decoder_params.get("num_layers", decoder_params.get("lstm_layers", 4)),
            dim_feedforward=decoder_params.get("dim_feedforward", 1024),
            dropout=decoder_params.get("dropout", 0.1),
            max_seq_length=decoder_params.get("max_seq_length", 141),
            pad_token_id=pad_token_id,
        )
        logger.info("Initialized %s model with vocab size %s", model_type, vocab_size)

    def encode(
        self,
        images: torch.Tensor,
        valid_widths: Optional[torch.Tensor] = None,
        valid_heights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.encoder(images, valid_widths=valid_widths, valid_heights=valid_heights)

    def forward(
        self,
        images: torch.Tensor,
        target_sequences: torch.Tensor,
        valid_widths: Optional[torch.Tensor] = None,
        valid_heights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory, mem_mask = self.encode(images, valid_widths, valid_heights)
        tgt = target_sequences[:, :-1]
        tgt_pad = tgt.eq(self.pad_token_id)
        return self.decoder(
            memory=memory,
            tgt_tokens=tgt,
            memory_key_padding_mask=mem_mask,
            tgt_key_padding_mask=tgt_pad,
        )

    def generate(
        self,
        images: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 141,
        beam_size: int = 1,
        length_penalty: float = 0.7,
        valid_widths: Optional[torch.Tensor] = None,
        valid_heights: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        memory, mem_mask = self.encode(images, valid_widths, valid_heights)
        if beam_size and beam_size > 1:
            return self._beam_search(
                memory, mem_mask, start_token_id, end_token_id, max_length, beam_size, length_penalty
            )
        return self._greedy_search(memory, mem_mask, start_token_id, end_token_id, max_length)

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
        length_penalty: float = 0.7,
        valid_widths: Optional[torch.Tensor] = None,
        valid_heights: Optional[torch.Tensor] = None,
    ) -> List[int]:
        if max_length is None:
            max_length = 141
        if beam_size is None:
            beam_size = 0
        sequences = self.generate(
            images=image,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            max_length=max_length,
            beam_size=max(beam_size, 1),
            length_penalty=length_penalty,
            valid_widths=valid_widths,
            valid_heights=valid_heights,
        )
        return sequences[0] if sequences else []

    def _greedy_search(
        self,
        memory: torch.Tensor,
        mem_mask: Optional[torch.Tensor],
        start_token_id: int,
        end_token_id: int,
        max_length: int,
    ) -> List[List[int]]:
        batch_size = memory.size(0)
        device = memory.device
        tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            tgt_pad = tokens.eq(self.pad_token_id)
            logits = self.decoder(
                memory=memory,
                tgt_tokens=tokens,
                memory_key_padding_mask=mem_mask,
                tgt_key_padding_mask=tgt_pad,
            )
            next_tokens = logits[:, -1].argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, self.pad_token_id), next_tokens)
            tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | next_tokens.eq(end_token_id)
            if torch.all(finished):
                break

        return [_strip_special(seq.tolist(), start_token_id, end_token_id, self.pad_token_id) for seq in tokens]

    def _beam_search(
        self,
        memory: torch.Tensor,
        mem_mask: Optional[torch.Tensor],
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        beam_size: int,
        length_penalty: float,
    ) -> List[List[int]]:
        batch_size, src_len, d_model = memory.shape
        device = memory.device
        vocab = self.vocab_size

        memory = (
            memory.unsqueeze(1)
            .expand(batch_size, beam_size, src_len, d_model)
            .reshape(batch_size * beam_size, src_len, d_model)
        )
        if mem_mask is not None:
            mem_mask = (
                mem_mask.unsqueeze(1)
                .expand(batch_size, beam_size, src_len)
                .reshape(batch_size * beam_size, src_len)
            )

        sequences = torch.full(
            (batch_size * beam_size, 1), start_token_id, dtype=torch.long, device=device
        )
        scores = torch.full((batch_size, beam_size), float("-inf"), device=device)
        scores[:, 0] = 0.0
        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            tgt_pad = sequences.eq(self.pad_token_id)
            logits = self.decoder(
                memory=memory,
                tgt_tokens=sequences,
                memory_key_padding_mask=mem_mask,
                tgt_key_padding_mask=tgt_pad,
            )
            log_probs = F.log_softmax(logits[:, -1], dim=-1).view(batch_size, beam_size, vocab)

            if finished.any():
                mask = finished.unsqueeze(-1)
                ninf = torch.full_like(log_probs, float("-inf"))
                finished_dist = ninf.clone()
                finished_dist[:, :, end_token_id] = 0.0
                log_probs = torch.where(mask, finished_dist, log_probs)

            cand_scores = scores.unsqueeze(-1) + log_probs
            cand_scores = cand_scores.view(batch_size, beam_size * vocab)
            topk_scores, topk_idx = torch.topk(cand_scores, beam_size, dim=-1)
            beam_idx = topk_idx // vocab
            token_idx = topk_idx % vocab

            base = (torch.arange(batch_size, device=device) * beam_size).unsqueeze(1)
            gather_idx = (base + beam_idx).reshape(-1)
            sequences = sequences[gather_idx]
            sequences = torch.cat([sequences, token_idx.reshape(-1, 1)], dim=1)
            scores = topk_scores
            finished = finished.view(batch_size * beam_size)[gather_idx].view(batch_size, beam_size)
            finished = finished | token_idx.eq(end_token_id)
            if torch.all(finished):
                break

        lengths = (sequences.ne(self.pad_token_id).sum(dim=1).view(batch_size, beam_size).float())
        lengths = torch.clamp(lengths, min=1.0)
        penalized = scores / ((5.0 + lengths) / 6.0).pow(length_penalty)
        penalized = penalized.masked_fill(~finished & (lengths < 2), float("-inf"))
        best = penalized.argmax(dim=-1)
        sequences = sequences.view(batch_size, beam_size, -1)
        result = []
        for b in range(batch_size):
            seq = sequences[b, best[b]].tolist()
            result.append(_strip_special(seq, start_token_id, end_token_id, self.pad_token_id))
        return result


def _strip_special(seq: List[int], start_id: int, end_id: int, pad_id: int) -> List[int]:
    cleaned = []
    for token in seq:
        if token in (start_id, pad_id):
            continue
        if token == end_id:
            break
        cleaned.append(token)
    return cleaned
