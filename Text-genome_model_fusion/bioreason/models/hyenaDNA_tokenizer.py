from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import os
from pathlib import Path
import json
from transformers.tokenization_utils_base import AddedToken

logger = logging.get_logger(__name__)

class Character_Tokenizer(PreTrainedTokenizer):
    """
    Character tokenizer for Hugging Face transformers.
    """
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        characters,
        model_max_length: int,
        padding_side: str = 'left',
        bos_token="[BOS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        **kwargs
    ):
        """
        Initialize the CharacterTokenizer.
        
        Args:
            characters (Sequence[str]): List of desired characters
            model_max_length (int): Model maximum sequence length
            padding_side (str): Padding side ('left' or 'right')
        """
        self.characters = characters
        self.model_max_length = model_max_length
        
        # Define special tokens with AddedToken
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False)
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False)
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False)

        # Build vocabulary
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return the vocab size of the tokenizer."""
        return len(self._vocab_str_to_int)

    def get_vocab(self) -> Dict[str, int]:
        """Return vocab as a dictionary."""
        return self._vocab_str_to_int.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize a string into characters."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to an id."""
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an id to a token."""
        return self._vocab_int_to_str.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence or a pair of sequence."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Retrieve sequence ids from a token list that has no special tokens added."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create a mask from the two sequences passed to be used in a sequence-pair classification task."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the tokenizer vocabulary to a directory."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        # Save vocabulary
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab_str_to_int, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)

    def get_config(self) -> Dict:
        """Get the tokenizer configuration."""
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        """Create a tokenizer from a configuration dictionary."""
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main tokenization method that handles batching and converts to tensors.
        """
        # Handle single string vs list of strings
        if isinstance(text, str):
            text = [text]
        
        # Set max length
        max_length = max_length or self.model_max_length
        
        # Tokenize all sequences
        input_ids_list = []
        for seq in text:
            # Tokenize
            tokens = self._tokenize(seq)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            
            # Add special tokens
            if add_special_tokens:
                token_ids = self.build_inputs_with_special_tokens(token_ids)
            
            # Truncate if needed
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                
            input_ids_list.append(token_ids)
        
        # Apply padding if needed
        if padding:
            max_len = max(len(ids) for ids in input_ids_list)
            
            # Create padded sequences and attention masks
            padded_input_ids = []
            attention_mask = []
            
            for ids in input_ids_list:
                # Apply padding based on padding_side
                padding_length = max_len - len(ids)
                if self.padding_side == "left":
                    padded_ids = [self.pad_token_id] * padding_length + ids
                    mask = [0] * padding_length + [1] * len(ids)
                else:
                    padded_ids = ids + [self.pad_token_id] * padding_length
                    mask = [1] * len(ids) + [0] * padding_length
                
                padded_input_ids.append(padded_ids)
                attention_mask.append(mask)
                
            input_ids_list = padded_input_ids
        else:
            # Create attention mask without padding
            attention_mask = [[1] * len(ids) for ids in input_ids_list]
        
        # Create result dictionary
        result = {"input_ids": input_ids_list}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
            
        # Convert to tensors if requested
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        elif return_tensors == "np":
            result = {k: np.array(v) for k, v in result.items()}
            
        return BatchEncoding(result, tensor_type=return_tensors)

    def batch_decode(
        self, 
        sequences: Union[List[int], List[List[int]], torch.Tensor, np.ndarray], 
        skip_special_tokens: bool = False, 
        **kwargs
    ) -> List[str]:
        """
        Decode a batch of token ids to strings.
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        elif isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        
        decoded_texts = []
        for seq in sequences:
            if not isinstance(seq, list):
                seq = [seq]
            
            # Convert ids to tokens
            tokens = [self._convert_id_to_token(id) for id in seq]
            
            # Remove special tokens if requested
            if skip_special_tokens:
                special_tokens = set(self.all_special_tokens)
                tokens = [token for token in tokens if token not in special_tokens]
            
            # Convert to string
            text = self.convert_tokens_to_string(tokens)
            decoded_texts.append(text)
        
        return decoded_texts
    
    def decode(
        self, 
        token_ids: Union[int, List[int], torch.Tensor, np.ndarray], 
        skip_special_tokens: bool = False, 
        **kwargs
    ) -> str:
        """
        Decode a single sequence of token ids to a string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Handle single id
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        return self.batch_decode([token_ids], skip_special_tokens, **kwargs)[0]


# Register the tokenizer with AutoTokenizer
def register_hyena_tokenizer():
    """Register the CharacterTokenizer with HuggingFace's AutoTokenizer."""
    AutoTokenizer.register("CharacterTokenizer", Character_Tokenizer)
    logger.info("hyena-CharacterTokenizer registered with AutoTokenizer")


if __name__ == "__main__":
    register_hyena_tokenizer()