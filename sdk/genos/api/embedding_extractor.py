#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import time
import logging
from typing import List, Union
from .health import BaseAPI
from ..exceptions import APIRequestError, ValidationError
import torch

# 设置日志
logger = logging.getLogger(__name__)


class EmbeddingExtractorAPI(BaseAPI):
    """Wrapper class for the Embedding Extraction API endpoints.
    
    This class handles requests to the embedding extraction endpoints,
    including single sequence and batch processing.
    """
    
    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30, config=None):
        """
        Initialize the EmbeddingExtractorAPI client.
        
        Args:
            session (requests.Session): Reusable HTTP session for API requests.
            base_url (str): Base URL of the Genos service.
            timeout (int, optional): Request timeout in seconds. Default is 30.
            config (GenosConfig, optional): Configuration object for endpoint management.
        """
        super().__init__(session, base_url, timeout)
        self.config = config
    
    def extract(self, sequence: str, model_name: str = "Genos-1.2B",
                pooling_method: str = "mean") -> Union[dict, List[dict]]:
        """
        Extracts a numerical embedding representation for a given nucleotide sequence.

        Args:
            sequence (str ): DNA sequence string .
            model_name (str, optional): Model name to use. Default is "Genos-1.2B".
                Options: "Genos-1.2B", "Genos-10B"
            pooling_method (str, optional): Pooling method. Default is "mean".
                Options: "mean", "max", "last", "none"

        Returns:
            dict:
                - "token_count": number of tokens
                - "embedding_shape": shape of embedding array
                - "embedding_dim": dimension of embedding
                - "embedding": embedding array (list)
        
        Raises:
            ValueError: If sequence is not a valid string or list.
            ValidationError: If parameters are invalid.
            APIRequestError: If the API request fails.
        
        Examples:
            >>> # Single sequence
            >>> result = embedding_api.extract("ATCGATCGATCG")
            >>> print(result['embedding_dim'])
            4096
        """
        # 判断是单个序列还是批量
        if isinstance(sequence, str):
            # 单个序列
            return self._extract_single(sequence, model_name, pooling_method)
        else:
            raise ValueError("sequence must be a string or list of strings")
    
    def _extract_single(self, sequence: str, model_name: str, pooling_method: str) -> dict:
        """内部方法：提取单个序列"""
        # Validate input
        if len(sequence) == 0:
            raise ValidationError("sequence cannot be empty")
        
        # Check sequence length limit (128K characters)
        if len(sequence) > 128000:
            raise ValidationError("sequence length cannot exceed 128,000 bases")
        
        # Prepare request
        # 从配置获取端点路径
        if self.config:
            endpoint = self.config.get_endpoint("embedding.extract")
        else:
            endpoint = ""  # 默认值（向后兼容）
        
        url = f"{self.base_url}{endpoint}"
        payload = {
            "sequence": sequence,
            "model_name": model_name,
            "pooling_method": pooling_method
        }
        
        # Start timing
        start_time = time.time()
        
        # Use the base class method for request handling with token validation
        data = self._make_request('POST', url, json=payload)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Check response status - new format has 'status' field
        # if data.get("status") != 200:
        #     error_msg = data.get("messages", "Unknown API error")
        #     raise APIRequestError(f"API request failed: {error_msg}")
        
        embedding = torch.tensor(data["result"]["embedding"])
        data["result"]["embedding"] = embedding
        # Print elapsed time to screen
        logger.info(f"⏱️  Embedding extraction completed in {elapsed_time:.4f}s "
                   f"(sequence_length={data.get('sequence_length', 'N/A')})")
        res = {}
        for key in ("result", "status", "message"):
            res[key] = data.get(key)
        return res