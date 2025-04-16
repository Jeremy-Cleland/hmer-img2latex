"""
Tokenizer for LaTeX formulas.
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch
from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class LaTeXTokenizer:
    """
    Tokenizer for LaTeX formulas.
    
    This tokenizer handles the conversion between LaTeX formula strings
    and token sequences, including handling of special tokens.
    """
    
    def __init__(
        self,
        special_tokens: Dict[str, str] = None,
        max_sequence_length: int = 150
    ):
        """
        Initialize the tokenizer.
        
        Args:
            special_tokens: Dictionary of special tokens
            max_sequence_length: Maximum sequence length
        """
        # Set default special tokens if not provided
        if special_tokens is None:
            self.special_tokens = {
                "PAD": "<PAD>",
                "START": "<START>",
                "END": "<END>",
                "UNK": "<UNK>",
            }
        else:
            self.special_tokens = special_tokens
        
        # Initialize vocabulary-related attributes
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.max_sequence_length = max_sequence_length
        
        # Initialize with special tokens
        self._init_special_tokens()
        
        logger.info(f"Initialized LaTeX tokenizer with max sequence length: {max_sequence_length}")
    
    def _init_special_tokens(self) -> None:
        """Initialize the vocabulary with special tokens."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens to the vocabulary
        for idx, token in enumerate(self.special_tokens.values()):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.vocab_size = len(self.token_to_id)
        
        # Save the indices of special tokens for easy access
        self.pad_token_id = self.token_to_id[self.special_tokens["PAD"]]
        self.start_token_id = self.token_to_id[self.special_tokens["START"]]
        self.end_token_id = self.token_to_id[self.special_tokens["END"]]
        self.unk_token_id = self.token_to_id[self.special_tokens["UNK"]]
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on a list of LaTeX formula texts.
        
        Args:
            texts: List of LaTeX formula texts
        """
        # Reset vocabulary
        self._init_special_tokens()
        
        # Count token frequencies
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)
        
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # Add tokens to vocabulary (skipping those already in vocabulary)
        for token, _ in sorted_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.id_to_token[self.vocab_size] = token
                self.vocab_size += 1
        
        # Check if max sequence length is sufficient
        max_found_length = max(len(text.split()) for text in texts)
        if max_found_length > self.max_sequence_length:
            logger.warning(
                f"Found sequences of length {max_found_length}, "
                f"which is longer than max_sequence_length ({self.max_sequence_length}). "
                "Consider increasing max_sequence_length."
            )
        
        logger.info(f"Fitted tokenizer on {len(texts)} texts. Vocabulary size: {self.vocab_size}")
    
    def fit_on_formulas_file(self, file_path: str) -> None:
        """
        Fit the tokenizer on a formulas file.
        
        Args:
            file_path: Path to the formulas file
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Formulas file not found: {file_path}")
        
        # Read formulas from file
        with open(file_path, "r", encoding="utf-8") as f:
            formulas = [line.strip() for line in f]
        
        # Add special tokens to formulas
        processed_formulas = [
            f"{self.special_tokens['START']} {formula} {self.special_tokens['END']}"
            for formula in formulas
        ]
        
        # Fit tokenizer on processed formulas
        self.fit(processed_formulas)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert a LaTeX formula text to a sequence of token IDs.
        
        Args:
            text: LaTeX formula text
            add_special_tokens: Whether to add START and END tokens
            
        Returns:
            List of token IDs
        """
        # Add special tokens if requested
        if add_special_tokens:
            text = f"{self.special_tokens['START']} {text} {self.special_tokens['END']}"
        
        # Split into tokens
        tokens = text.split()
        
        # Convert tokens to IDs
        ids = [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a sequence of token IDs to a LaTeX formula text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            LaTeX formula text
        """
        # Get special token IDs
        special_token_ids = [
            self.token_to_id[token] for token in self.special_tokens.values()
        ] if skip_special_tokens else []
        
        # Convert IDs to tokens, skipping special tokens if requested
        tokens = [
            self.id_to_token.get(idx, self.special_tokens["UNK"])
            for idx in ids
            if idx not in special_token_ids or not skip_special_tokens
        ]
        
        # Join tokens into a string
        text = " ".join(tokens)
        
        return text
    
    def encode_batch(
        self, 
        texts: List[str], 
        add_special_tokens: bool = False,
        padding: bool = True,
        truncation: bool = True
    ) -> torch.Tensor:
        """
        Convert a batch of LaTeX formula texts to a batch of token ID sequences.
        
        Args:
            texts: List of LaTeX formula texts
            add_special_tokens: Whether to add START and END tokens
            padding: Whether to pad sequences to max_sequence_length
            truncation: Whether to truncate sequences longer than max_sequence_length
            
        Returns:
            Tensor of token IDs of shape (batch_size, seq_length)
        """
        # Encode each text
        encoded_texts = [self.encode(text, add_special_tokens) for text in texts]
        
        # Truncate if needed
        if truncation:
            encoded_texts = [
                ids[:self.max_sequence_length] 
                for ids in encoded_texts
            ]
        
        # Pad if needed
        if padding:
            encoded_texts = [
                ids + [self.pad_token_id] * (self.max_sequence_length - len(ids))
                for ids in encoded_texts
            ]
        
        # Convert to tensor
        batch_tensor = torch.tensor(encoded_texts)
        
        return batch_tensor
    
    def decode_batch(
        self, 
        batch_ids: torch.Tensor, 
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Convert a batch of token ID sequences to a batch of LaTeX formula texts.
        
        Args:
            batch_ids: Tensor of token IDs of shape (batch_size, seq_length)
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            List of LaTeX formula texts
        """
        # Convert tensor to list of lists
        batch_ids_list = batch_ids.tolist()
        
        # Decode each sequence
        decoded_texts = [
            self.decode(ids, skip_special_tokens) 
            for ids in batch_ids_list
        ]
        
        return decoded_texts
    
    def save(self, file_path: str) -> None:
        """
        Save the tokenizer vocabulary to a file.
        
        Args:
            file_path: Path to save the vocabulary to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save vocabulary (token -> id)
        save_dict = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
            "max_sequence_length": self.max_sequence_length,
        }
        
        torch.save(save_dict, file_path)
        logger.info(f"Saved tokenizer to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'LaTeXTokenizer':
        """
        Load a tokenizer from a file.
        
        Args:
            file_path: Path to load the vocabulary from
            
        Returns:
            Loaded tokenizer
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
        
        # Load vocabulary
        save_dict = torch.load(file_path)
        
        # Create tokenizer
        tokenizer = cls(
            special_tokens=save_dict["special_tokens"],
            max_sequence_length=save_dict["max_sequence_length"]
        )
        
        # Restore vocabulary
        tokenizer.token_to_id = save_dict["token_to_id"]
        tokenizer.vocab_size = len(tokenizer.token_to_id)
        
        # Rebuild id_to_token mapping
        tokenizer.id_to_token = {
            idx: token for token, idx in tokenizer.token_to_id.items()
        }
        
        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.special_tokens["PAD"]]
        tokenizer.start_token_id = tokenizer.token_to_id[tokenizer.special_tokens["START"]]
        tokenizer.end_token_id = tokenizer.token_to_id[tokenizer.special_tokens["END"]]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.special_tokens["UNK"]]
        
        logger.info(f"Loaded tokenizer from {file_path} with vocabulary size: {tokenizer.vocab_size}")
        return tokenizer