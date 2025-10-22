#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import logging
from .health import BaseAPI
from ..exceptions import APIRequestError

logger = logging.getLogger(__name__)

CHROMOSOMES = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

ALLOWED_RANGES = {
    'hg38': {'chr1': (10001, 248946422), 'chr2': (10001, 242183529), 'chr3': (10001, 198235559), 'chr4': (10001, 190204555), 'chr5': (10001, 181478259), 'chr6': (60001, 170745979), 'chr7': (10001, 159335973), 'chr8': (60001, 145078636), 'chr9': (10001, 138334717), 'chr10': (10001, 133787422), 'chr11': (60001, 135076622), 'chr12': (10001, 133265309), 'chr13': (16000001, 114354328), 'chr14': (16000001, 106883718), 'chr15': (17000001, 101981189), 'chr16': (10001, 90228345), 'chr17': (60001, 83247441), 'chr18': (10001, 80263285), 'chr19': (60001, 58607616), 'chr20': (60001, 64334167), 'chr21': (5010001, 46699983), 'chr22': (10510001, 50808468), 'chrX': (10001, 156030895), 'chrY': (10001, 57217415)}, 
    'hg19': {'chr1': (10001, 249240621), 'chr2': (10001, 243189373), 'chr3': (60001, 197962430), 'chr4': (10001, 191044276), 'chr5': (10001, 180905260), 'chr6': (60001, 171055067), 'chr7': (10001, 159128663), 'chr8': (10001, 146304022), 'chr9': (10001, 141153431), 'chr10': (60001, 135524747), 'chr11': (60001, 134946516), 'chr12': (60001, 133841895), 'chr13': (19020001, 115109878), 'chr14': (19000001, 107289540), 'chr15': (20000001, 102521392), 'chr16': (60001, 90294753), 'chr17': (1, 81195210), 'chr18': (10001, 78017248), 'chr19': (60001, 59118983), 'chr20': (60001, 62965520), 'chr21': (9411194, 48119895), 'chr22': (16050001, 51244566), 'chrX': (60001, 155260560), 'chrY': (10001, 59363566)}
    }


def check_chromosome(chrom: str) -> bool:
    """
    Check if the chromosome is valid.
    """
    return chrom in CHROMOSOMES


def check_position(assembly: str, chrom: str, pos: int) -> bool:
    """
    Check if the position is valid.
    """
    return ALLOWED_RANGES[assembly][chrom][0] <= pos <= ALLOWED_RANGES[assembly][chrom][1]



class VariantPredictorAPI(BaseAPI):
    """Wrapper class for the Variant Prediction API endpoint.

    This class handles requests to the `/predict` endpoint for variant
    pathogenicity prediction. It validates input and raises structured
    SDK exceptions on errors.
    """

    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30):
        """
        Initialize the VariantPredictorAPI client.

        Args:
            session (requests.Session): Reusable HTTP session for API requests.
            base_url (str): Base URL of the Genos service.
            timeout (int, optional): Request timeout in seconds. Default is 30.
        """
        super().__init__(session, base_url, timeout)

    def predict(self, assembly: str, chrom: str, pos: int, ref: str, alt: str) -> dict:
        """
        Predict the pathogenicity of a genetic variant.
        
        Args:
            assembly (str): Reference genome version, allowed values: 'hg38' or 'hg19'.
            chrom (str): Chromosome, e.g., 'chr6'.
            pos (int): Position, 1-based coordinate.
            ref (str): Reference allele, single letter or sequence.
            alt (str): Alternate allele, single letter or sequence.

        Returns:
            dict: Prediction result, typically containing:
                - "variant": input variant
                - "prediction": "Pathogenic" or "Benign"
                - "score_Benign": float
                - "score_Pathogenic": float

        Raises:
            ValueError: If any argument is invalid.
            APIRequestError: If the API request fails or the server returns an error.
        """
        # Validate assembly
        allowed_assemblies = ["hg38", "hg19"]
        if assembly not in allowed_assemblies:
            raise ValueError(f"assembly must be one of {allowed_assemblies}, but got '{assembly}'")
        
        # Validate chromosome
        if not check_chromosome(chrom):
            raise ValueError(f"Invalid chromosome: {chrom}. Allowed: {', '.join(CHROMOSOMES)}")
        
        # Validate position
        if not isinstance(pos, int) or not check_position(assembly, chrom, pos):
            raise ValueError(f"Position {pos} not allowed on {assembly} {chrom}")
        
        # Validate reference and alternate alleles
        valid_bases = set('ATCGN')
        if not isinstance(ref, str) or len(ref) == 0 or any(base.upper() not in valid_bases for base in ref):
            raise ValueError("ref allele must be a non-empty string composed only of nucleotides A, T, C, G, or N")
        
        if not isinstance(alt, str) or len(alt) == 0 or any(base.upper() not in valid_bases for base in alt):
            raise ValueError("alt allele must be a non-empty string composed only of nucleotides A, T, C, G, or N")
        
        # Prepare payload with individual parameters
        url = f"{self.base_url}"
        payload = {
            "assembly": assembly,
            "chrom": chrom,
            "pos": pos,
            "ref": ref.upper(),
            "alt": alt.upper()
        }

        start_time = time.time()

        # Use the base class method for request handling with token validation
        data = self._make_request('POST', url, json=payload)

        elapsed_time = time.time() - start_time

        # Check response status - new format has 'status' field
        # if data.get("status") != 200:
        #     error_msg = data.get("messages", "Unknown API error")
        #     raise APIRequestError(f"API request failed: {error_msg}")

        logger.info(f"Variant prediction completed in {elapsed_time:.4f}s ")

        res = {}
        for key in ("result", "status", "message"):
            res[key] = data.get(key)
        return res
