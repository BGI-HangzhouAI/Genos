#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .version import __version__
from .client import GenosClient
from .exceptions import PackageUpdateRequired


def create_client(token: str = None, timeout: int = 30, variant_model: str = None, embedding_model: str = None) -> GenosClient:
    """
    Factory function to create a GenosClient instance.

    Example:
        >>> from genos import create_client
        >>> client = create_client(token="your_token")
        >>> result = client.variant_predict("hg38", "chr6", 51484075, "T", "G")

    Args:
        token (str, optional): API access token. Default: None.
        base_url (str, optional): Base URL of the Genos service.
            Defaults to environment variable GENOS_API_URL or "http://localhost:5000".
        timeout (int, optional): Request timeout in seconds. Default: 30.

    Returns:
        GenosClient: A configured client instance ready to make API calls.
    """
    return GenosClient(token=token, timeout=timeout, variant_model=variant_model, embedding_model=embedding_model)


__all__ = ["GenosClient", "PackageUpdateRequired", "create_client", "__version__"]

