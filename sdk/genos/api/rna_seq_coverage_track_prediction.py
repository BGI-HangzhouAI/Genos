#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
import time
import logging
from .health import BaseAPI
from ..exceptions import APIRequestError

CHROMOSOMES = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
               'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']

# 设置日志
logger = logging.getLogger(__name__)
# Dictionary mapping chromosome names to their lengths (GRCh38/hg38)
CHROM_LENGTHS = {
    'chr1': 248956422,
    'chr2': 242193529,
    'chr3': 198295559,
    'chr4': 190214555,
    'chr5': 181538259,
    'chr6': 170805979,
    'chr7': 159345973,
    'chr8': 145138636,
    'chr9': 138394717,
    'chr10': 133797422,
    'chr11': 135086622,
    'chr12': 133275309,
    'chr13': 114364328,
    'chr14': 107043718,
    'chr15': 101991189,
    'chr16': 90338345,
    'chr17': 83257441,
    'chr18': 80373285,
    'chr19': 58617616,
    'chr20': 64444167,
    'chr21': 46709983,
    'chr22': 50818468,
    'chrX': 156040895,
    'chrY': 57227415,
}


def check_chromosome(chrom: str) -> bool:
    """
    Check if the chromosome is valid.
    """
    return chrom in CHROMOSOMES


def get_start_position_range(chrom: str, gen_length: int = 32000):
    """
    Return the allowed range for start_pos for the given chromosome.
    Allowed: 1 ~ (chrom_length - 32000), inclusive.

    Raises:
        ValueError: If the chromosome is not recognized.
    """
    if chrom not in CHROM_LENGTHS:
        raise ValueError(f"Unknown chromosome: {chrom}")
    chrom_length = CHROM_LENGTHS[chrom]
    # Ensure the end is not less than 1
    end_pos = max(1, chrom_length - gen_length)
    return end_pos  # inclusive of end_pos


def check_valid_parameters(chrom: str, start_pos: int, gen_length: int = 32000):
    """
    Check if the parameters are valid.
    Args:
        chrom (str): Chromosome name.
        start_pos (int): Start position.
        gen_length (int): Generation length.
    Returns:
        True if the parameters are valid, False otherwise.
    """
    # check chromosome
    if not check_chromosome(chrom):
        raise ValueError(f"Invalid chromosome: {chrom}. Allowed: {', '.join(CHROMOSOMES)}")
    # check start position
    if not isinstance(start_pos, int) or start_pos <= 0:
        raise ValueError(f"Start position must be a positive integer, but got {start_pos}")
    end_pos_max = get_start_position_range(chrom, gen_length)
    if start_pos > end_pos_max:
        raise ValueError(f"Start position must be less than {end_pos_max}, but got {start_pos}")
    return True


class RNASeqCoverageTrackPredictionAPI(BaseAPI):
    """Client for the RNA-seq Coverage Track Prediction service in GeneOS.

    This class provides access to RNA-seq Coverage Track Prediction
    models, allowing users to predict the coverage track of an RNA-seq
    experiment based on the genomic coordinates.

    """

    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30):
        """
        Initialize the RNA-seq Coverage Track Prediction API module.

        Args:
            session (requests.Session): Shared HTTP session from GenosClient.
            base_url (str): Endpoint URL for RNA-seq Coverage Track Prediction service.
            timeout (int, optional): Request timeout (seconds). Default: 30.
        """
        super().__init__(session, base_url, timeout)

    def predict(self, chrom: str, start_pos: int) -> dict:
        """
        Predict the coverage track of an RNA-seq experiment based on the genomic coordinates.

        Args:
            chrom (str): Chromosome name.
            start_pos (int): Start position.

        Returns:
            dict: A JSON response from the RNA-seq Coverage Track Prediction model containing the predicted coverage track.

        Raises:
            ValueError: If the parameters are invalid.
            APIRequestError: If the API request fails.
        """
        # check parameters
        check_valid_parameters(chrom, start_pos)

        payload = {"chrom": chrom, "start_pos": start_pos}

        start_time = time.time()
        # Use the base class method for request handling with token validation
        data = self._make_request('POST', self.base_url, json=payload)
        elapsed_time = time.time() - start_time
        print(f"RNA-seq coverage track prediction completed in {elapsed_time:.4f}s")
        res = {}
        for key in ("result", "status", "message"):
            res[key] = data.get(key)
        return res


