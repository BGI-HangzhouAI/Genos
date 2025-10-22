#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


class GenosConfig:
    """
    Configuration class for the GeneOS API client.

    Supports fully independent URLs for each analysis module (e.g., mutation, embedding, RNA),
    with optional environment variable overrides.

    Environment variable overrides:
        - GENOS_API_URL_MUTATION
        - GENOS_API_URL_EMBEDDING
        - GENOS_API_URL_RNA
        - GENOS_API_TOKEN

    Example:
        >>> cfg = GenosConfig()
        >>> cfg.get_api_url("mutation")
        'https://mutation.geneos.ai/v1/predict'
    """

    DEFAULT_API_MAP = {
        "variant": "https://cloud.stomics.tech/api/aigateway/genos/variant_predict",
        "embedding": "https://cloud.stomics.tech/api/aigateway/genos/embedding",
        "rna": "https://cloud.stomics.tech/api/aigateway/genos/rna_seq_coverage_track",
    }

    def __init__(self, token=None, timeout=30, api_map=None):
        """
        Initialize the configuration.

        Args:
            token (str, optional): API token or access key.
                Defaults to environment variable GENOS_API_TOKEN.
            timeout (int, optional): Request timeout (seconds). Default is 30.
            api_map (dict, optional): Custom API endpoint mapping per module.
                e.g., {"mutation": "https://myhost/mut", "embedding": "http://127.0.0.1:8000/embed"}
        """
        self.token = token or os.getenv("GENOS_API_TOKEN")
        self.timeout = timeout

        # 合并默认与自定义映射
        self.api_map = {**self.DEFAULT_API_MAP, **(api_map or {})}

        # 允许通过环境变量覆盖
        for key in self.api_map.keys():
            env_key = f"GENOS_API_URL_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val:
                self.api_map[key] = env_val

    def get_api_url(self, module_name: str) -> str:
        """
        Get the API URL for a specific analysis module.

        Args:
            module_name (str): Analysis module name (e.g., 'mutation', 'embedding', 'rna').

        Returns:
            str: Full API URL.

        Raises:
            KeyError: If the module is not defined in api_map.
        """
        if module_name not in self.api_map:
            raise KeyError(f"Unknown module: {module_name}. Available: {list(self.api_map.keys())}")
        return self.api_map[module_name]

    def set_api_url(self, module_name: str, url: str):
        """Dynamically update or add a module URL at runtime."""
        self.api_map[module_name] = url

    def list_modules(self):
        """Return all configured module names."""
        return list(self.api_map.keys())

    def __repr__(self):
        return f"<GenosConfig modules={list(self.api_map.keys())}, timeout={self.timeout}>"
