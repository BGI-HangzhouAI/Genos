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

# 网关注册名用下划线。调用方仍传 Genos-1.2B。
REQUEST_MODEL_NAMES = {
    "Genos-1.2B": "Genos-1_2B",
}


class EmbeddingExtractorAPI(BaseAPI):
    """Wrapper class for the Embedding Extraction API endpoints.
    
    This class handles requests to the embedding extraction endpoints,
    including single sequence and batch processing.
    """
    
    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30, model: str = "Genos-1.2B", config=None):
        """
        Initialize the EmbeddingExtractorAPI client.
        
        Args:
            session (requests.Session): Reusable HTTP session for API requests.
            base_url (str): Base URL of the Genos service.
            timeout (int, optional): Request timeout in seconds. Default is 30.
            model (str, optional): Default value for the request field ``model``
                when ``model_name`` is omitted. Default is "Genos-1.2B".
            config (GenosConfig, optional): Unused. The base URL is already the full endpoint.
        """
        super().__init__(session, base_url, timeout)
        self.model = model
        self.config = config
    
    def extract(self, sequence: str, model_name: str = None,
                pooling_method: str = "mean") -> Union[dict, List[dict]]:
        """
        Extracts a numerical embedding representation for a given nucleotide sequence.

        ``model_name`` is sent unchanged. The gateway field ``model`` uses the
        registered name, so ``Genos-1.2B`` is sent as ``model="Genos-1_2B"``.

        Args:
            sequence (str ): DNA sequence string .
            model_name (str, optional): Sent as ``model_name``. Default is "Genos-1.2B".
                Options: "Genos-1.2B", "Genos-10B", "Genos-10B-v2".
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

        weight = model_name or self.model
        url = self.base_url
        payload = {
            "model": REQUEST_MODEL_NAMES.get(weight, weight),
            "model_name": weight,
            "sequence": sequence,
            "pooling_method": pooling_method,
        }
        
        # Start timing
        start_time = time.time()
        
        # Use the base class method for request handling with token validation
        data = self._make_request('POST', url, json=payload)
        body_status = data.get("status") if isinstance(data, dict) else None
        if body_status not in (None, 200, "200"):
            message = None
            if isinstance(data, dict):
                message = data.get("messages") or data.get("message")
            raise APIRequestError(
                f"API request failed: {message or 'embedding request failed'}",
                body_status,
            )
        
        elapsed_time = time.time() - start_time
        embedding = torch.tensor(data["result"]["embedding"])
        data["result"]["embedding"] = embedding
        # Print elapsed time to screen
        logger.info(f"⏱️  Embedding extraction completed in {elapsed_time:.4f}s "
                   f"(sequence_length={data.get('sequence_length', 'N/A')})")
        return {
            "result": data.get("result"),
            "status": data.get("status", 200),
            "message": data.get("message") or data.get("messages"),
        }