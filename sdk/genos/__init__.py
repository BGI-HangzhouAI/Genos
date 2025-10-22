#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .client import GenosClient


def create_client(token: str = None, timeout: int = 30) -> GenosClient:
    """
    Factory function to create a GenosClient instance.

    Example:
        >>> from genos import create_client
        >>> client = create_client(token="your_token")
        >>> result = client.variant_predict("hg19", "chr6", 51484075, "T", "G")

    Args:
        token (str, optional): API access token. Default: None.
        base_url (str, optional): Base URL of the Genos service.
            Defaults to environment variable GENOS_API_URL or "http://localhost:5000".
        timeout (int, optional): Request timeout in seconds. Default: 30.

    Returns:
        GenosClient: A configured client instance ready to make API calls.
    """
    return GenosClient(token=token, timeout=timeout)


__all__ = ["GenosClient", "create_client"]

