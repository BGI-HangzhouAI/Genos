import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import AutoModelForMaskedLM, AutoTokenizer,AutoModelForCausalLM
from bioreason.models.hyenaDNA import HyenaDNAPreTrainedModel
from bioreason.models.hyenaDNA_tokenizer import Character_Tokenizer
from bioreason.models.evo2_tokenizer import Evo2Tokenizer


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        # Use PyTorch's built-in multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, embeddings, attention_mask=None):
        # Expand query to batch size
        batch_size = embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        # Create key padding mask from attention mask if provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # Convert to boolean mask where True means ignore
        
        # Apply attention: query attends to embeddings
        context, _ = self.attention(
            query=query,                  # [batch_size, 1, hidden_size]
            key=embeddings,               # [batch_size, seq_len, hidden_size]
            value=embeddings,             # [batch_size, seq_len, hidden_size]
            key_padding_mask=key_padding_mask
        )
        
        # Squeeze out the singleton dimension
        return context.squeeze(1)         # [batch_size, hidden_size]


class DNAClassifierModel(nn.Module):
    """
    A simple classifier that uses a DNA model with a classification head.
    """

    def __init__(
        self,
        dna_model_name: str,
        cache_dir: str = None,
        max_length_dna: int = 4096,
        num_classes: int = 2,  # Binary classification by default
        dna_is_evo2: bool = False,
        dna_embedding_layer: str = None,
        train_just_classifier: bool = True
    ):
        """
        Initialize the DNAClassifierModel.

        Args:
            dna_model_name (str): Name of the DNA model to use
            cache_dir (str): Directory to cache models
            max_length_dna (int): Maximum sequence length
            num_classes (int): Number of output classes
            dna_is_evo2: Whether the DNA model is Evo2. Defaults to False
            dna_embedding_layer: Name of the layer to use for the Evo2 model. Defaults to None
            train_just_classifier: Whether to train just the classifier. Defaults to True
        """
        super().__init__()

        self.dna_model_name = dna_model_name
        self.cache_dir = cache_dir
        self.max_length_dna = max_length_dna
        self.num_classes = num_classes
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer
        self.train_just_classifier = train_just_classifier
        self.is_hyenadna = dna_model_name.lower().find("hyenadna")>=0

        # Load the DNA model and tokenizer
        print('if evo2:',self.dna_is_evo2)
        if not self.dna_is_evo2:
            try:
                print(f'is_hyenadna: {self.is_hyenadna}')
                
                if self.is_hyenadna:
                    
                    self.dna_model = HyenaDNAPreTrainedModel.from_pretrained(
                        cache_dir,dna_model_name,download=False,device='cuda'
                    )
                    self.dna_tokenizer = Character_Tokenizer(
                        characters=['A', 'C', 'G', 'T', 'N'],
                        # add DNA characters, N is uncertain
                        model_max_length=max_length_dna + 2,  # to account for special tokens, like EOS
                        # we handle special tokens elsewhere
                        padding_side='left', # since HyenaDNA is causal, we pad on the left
                    ) 
                    with open(os.path.join(os.path.join(cache_dir,dna_model_name), "config.json")) as f:
                        self.dna_config = json.load(f)
                        self.dna_hidden_size = self.dna_config['d_model']
                else:
                    self.dna_model = AutoModelForMaskedLM.from_pretrained(
                        dna_model_name,  trust_remote_code=True
                    )
                    self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
                    self.dna_config = self.dna_model.config
                    self.dna_hidden_size = self.dna_config.hidden_size
            except Exception as e:
                self.dna_model = AutoModelForCausalLM.from_pretrained(
                    dna_model_name,  trust_remote_code=True
                )
                self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
                self.dna_config = self.dna_model.config  
                self.dna_hidden_size = self.dna_config.hidden_size          

        else:
            from evo2 import Evo2
            self.dna_model = Evo2(dna_model_name,local_path=cache_dir)
            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_config = self.dna_model.model.config
            self.dna_embedding_layer = self.dna_embedding_layer
            self.dna_hidden_size = self.dna_config.hidden_size
# Get hidden size from model config
        self.hidden_size = self.dna_hidden_size

        # Add the self-attention pooling module
        self.pooler = SelfAttentionPooling(self.hidden_size)

        # Create classification head that takes concatenated embeddings from both sequences
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )

        self.max_length_dna = max_length_dna

    def get_dna_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Get DNA embedding for a single DNA sequence using self-attention pooling.

        Args:
            input_ids: DNA tokenized sequence
            attention_mask: DNA tokenized sequence attention mask

        Returns:
            torch.Tensor: Tensor containing the self-attention pooled DNA embedding
        """
        # Add batch dimension if not present
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
        
        # Handle attention mask - create if not provided or add batch dimension
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
        
        # Get embeddings from DNA model
        with torch.set_grad_enabled(not self.train_just_classifier):  # Enable gradients for fine-tuning

            if self.dna_is_evo2 and self.dna_embedding_layer is not None:  # Evo2 model
                # Get embeddings from the specific layer in Evo2
                _, embeddings = self.dna_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[self.dna_embedding_layer]
                )
                
                # Get embeddings for the specified layer
                hidden_states = embeddings[self.dna_embedding_layer]
            
            else:
                # Get embeddings from the last hidden state
                outputs = self.dna_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get the last hidden state
                if self.is_hyenadna:
                    hidden_states = outputs
                else:
                # Get the last hidden state
                    hidden_states = outputs.hidden_states[-1] 
        
        # Apply self-attention pooling to get a weighted representation
        sequence_embedding = self.pooler(hidden_states, attention_mask)
        return sequence_embedding.squeeze(0)

    def forward(
        self, ref_ids=None, alt_ids=None, ref_attention_mask=None, alt_attention_mask=None
    ):
        """
        Forward pass of the model.

        Args:
            ref_ids: Reference sequence token IDsself.dna_model
            alt_ids: Alternate sequence token IDsself.dna_model
            ref_attention_mask: Reference sequence attention maskself.dna_model
            alt_attention_mask: Alternate sequence attention maskself.dna_model

        Returns:
            torch.Tensor: Classification logits
        """
        batch_size = ref_ids.shape[0] if ref_ids is not None else alt_ids.shape[0]

        if batch_size is None:
            raise ValueError("Either token IDs must be provided")

        ref_embeddings = []
        alt_embeddings = []

        # Process each example in the batch
        for i in range(batch_size):

            # Get sequence embeddings
            ref_embed = self.get_dna_embedding(ref_ids[i], ref_attention_mask[i])
            alt_embed = self.get_dna_embedding(alt_ids[i], alt_attention_mask[i])
            ref_embeddings.append(ref_embed)
            alt_embeddings.append(alt_embed)

        # Stack embeddings
        ref_embeddings = torch.stack(ref_embeddings)
        alt_embeddings = torch.stack(alt_embeddings)

        # Concatenate ref and alt embeddings
        combined_embeddings = torch.cat([ref_embeddings, alt_embeddings], dim=1)

        # Pass through classifier
        logits = self.classifier(combined_embeddings)

        return logits