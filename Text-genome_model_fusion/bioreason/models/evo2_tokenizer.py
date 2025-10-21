from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# Register the tokenizer with AutoTokenizer
from transformers.models.auto import AutoTokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

class Evo2Tokenizer(PreTrainedTokenizer):
    """
    Tokenizer for Evo2 models - wraps the CharLevelTokenizer to be compatible with HuggingFace.
    """
    vocab_files_names = {}  # No vocab files needed
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        evo2_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        **kwargs
    ):
        """
        Initialize the Evo2Tokenizer.
        
        Args:
            evo2_tokenizer: The Evo2 CharLevelTokenizer to wrap
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
            unk_token: Unknown token
        """
        self.evo2_tokenizer = evo2_tokenizer
        
        # Map special tokens to Evo2 tokenizer's special token IDs
        self._pad_token = pad_token
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._unk_token = unk_token
        
        # Initialize with special tokens
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs
        )
        
        # Set token IDs from Evo2 tokenizer
        self.pad_token_id = self.evo2_tokenizer.pad_id
        self.eos_token_id = self.evo2_tokenizer.eos_id
        
    @property
    def vocab_size(self) -> int:
        """Return the vocab size of the tokenizer."""
        return self.evo2_tokenizer.vocab_size
    
    def get_vocab(self) -> Dict:
        """Return vocab as a dictionary."""
        # Evo2 CharLevelTokenizer doesn't have a traditional vocab dict
        # Create a simple mapping of ASCII codes to tokens
        return {chr(i): i for i in range(self.vocab_size)}
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize a string using the Evo2 tokenizer."""
        return [chr(int(token)) for token in self.evo2_tokenizer.tokenize(text)]
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to an id using the Evo2 tokenizer."""
        # Since tokens are just characters, convert to their ASCII value
        return ord(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an id to a token using the Evo2 tokenizer."""
        # Convert ASCII value back to character
        return chr(index)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        return "".join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """No vocabulary to save for Evo2Tokenizer, so just return an empty tuple."""
        return ()
    
    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Main tokenization method that handles batching and converts to tensors.
        """
        # Handle single string vs list of strings
        if isinstance(text, str):
            text = [text]
            
        # Tokenize all sequences - note: tokenizer only accepts strings, not lists
        input_ids_list = []
        for seq in text:
            # Tokenize and convert numpy.uint8 to Python integers
            tokens = [int(token) for token in self.evo2_tokenizer.tokenize(seq)]
            
            # Truncate if needed
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
                
            input_ids_list.append(tokens)
        
        # Apply padding if needed
        if padding:
            if False:#max_length:
                max_len = max_length
            else:
                max_len = max(len(ids) for ids in input_ids_list)
            
            # Create padded sequences and attention masks
            padded_input_ids = []
            attention_mask = []
            
            for ids in input_ids_list:
                # Apply left padding (pad on the left)
                padding_length = max_len - len(ids)
                padded_ids = [self.pad_token_id] * padding_length + ids
                mask = [0] * padding_length + [1] * len(ids)
                
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
            
        # Return a BatchEncoding object rather than a plain dictionary
        return BatchEncoding(
            data=result,
            tensor_type=return_tensors,
            prepend_batch_axis=False,  # Already handled in our tensor creation
            encoding=None  # No encoding info from Evo2's tokenizer
        )
    
    def batch_decode(
        self, 
        sequences: Union[List[int], List[List[int]], torch.Tensor], 
        skip_special_tokens: bool = False, 
        **kwargs
    ) -> List[str]:
        """
        Decode a batch of token ids to strings.
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        return self.evo2_tokenizer.detokenize_batch(sequences)
    
    def decode(
        self, 
        token_ids: Union[int, List[int], torch.Tensor], 
        skip_special_tokens: bool = False, 
        **kwargs
    ) -> str:
        """
        Decode a single sequence of token ids to a string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Single sequence
        if not isinstance(token_ids, list) or not token_ids or not isinstance(token_ids[0], (list, torch.Tensor)):
            return self.evo2_tokenizer.detokenize(token_ids)
        
        # Batch with one item
        return self.batch_decode(token_ids, skip_special_tokens, **kwargs)[0]


# Register the tokenizer - you'll need to do this when your script loads
# You might want to put this in your __init__.py file
def register_evo2_tokenizer():
    """Register the Evo2Tokenizer with HuggingFace's AutoTokenizer."""
    
    # This will register the tokenizer so AutoTokenizer.from_pretrained knows about it
    AutoTokenizer.register("evo2", Evo2Tokenizer)
    
    # If you have a config class, you would also register that
    # from transformers.models.auto import AutoConfig
    # AutoConfig.register("evo2", Evo2Config)
    
    print("Evo2Tokenizer registered with AutoTokenizer")


if __name__ == "__main__":
    register_evo2_tokenizer()