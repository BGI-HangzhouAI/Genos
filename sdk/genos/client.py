#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from .config import GenosConfig
from .api.variant_predictor import VariantPredictorAPI
from .api.embedding_extractor import EmbeddingExtractorAPI
from .api.rna_seq_coverage_track_prediction import RNASeqCoverageTrackPredictionAPI
from .api.health import HealthAPI


class GenosClient:
    """GeneOS API unified client.

    Provides a high-level SDK interface for users to access
    multiple biological analysis models (mutation, embedding, RNA).

    Example:
        >>> from genos import create_client
        >>> client = create_client(token="your_token")
        >>> client.variant_predict("hg19", "chr6", 51484075, "T", "G")
        >>> client.get_embedding("ATGC...")
        >>> client.rna_coverage_track_pred(chrom="chr6", start_pos=51484075)
    """

    def __init__(self, token=None, timeout=30, api_map=None):
        """
        Initialize the GeneOS API client.

        Args:
            token (str, optional): API authentication token.
                Defaults to environment variable GENOS_API_TOKEN.
            timeout (int, optional): Request timeout (seconds). Default 30.
            api_map (dict, optional): Custom mapping for each API module URL.
        """
        # Load full configuration
        self.config = GenosConfig(token=token, timeout=timeout, api_map=api_map)

        # Setup a shared HTTP session
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        if self.config.token:
            self.session.headers.update({"Authorization": f"Bearer {self.config.token}"})
        else:
            self.session.headers.update({"Authorization": f"Bearer <your_api_key>"})  # TODO: for test

        # Initialize all API modules

        self.variant = VariantPredictorAPI(
            self.session, self.config.get_api_url("variant"), self.config.timeout
        )
        self.embedding = EmbeddingExtractorAPI(
            self.session, self.config.get_api_url("embedding"), self.config.timeout
        )
        self.rna = RNASeqCoverageTrackPredictionAPI(
            self.session, self.config.get_api_url("rna"), self.config.timeout
        )

    def variant_predict(self, assembly: str, chrom: str, pos: int, ref: str, alt: str):
        """
        Predicts the functional or pathogenic effect of a given genetic variant.

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
        """
        return self.variant.predict(assembly, chrom, pos, ref, alt)

    def get_embedding(self, sequence: str, model_name: str = "Genos-1.2B", pooling_method: str = "mean"):
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
                - "sequence": input sequence
                - "sequence_length": length of sequence
                - "token_count": number of tokens
                - "embedding_shape": shape of embedding array
                - "embedding_dim": dimension of embedding
                - "pooling_method": pooling method used
                - "model_type": model type used
                - "embedding": embedding array (list)

        Raises:
            ValueError: If sequence is not a valid string or list.
            ValidationError: If parameters are invalid.
            APIRequestError: If the API request fails.
        """
        return self.embedding.extract(sequence, model_name, pooling_method)

    def rna_coverage_track_pred(self, chrom: str, start_pos: int):
        """
        Predicts RNA-seq coverage track based on genomic coordinates.

        This method provides access to RNA-seq Coverage Track Prediction
        models, allowing users to predict the coverage track of an RNA-seq
        experiment based on the genomic coordinates.

        Args:
            chrom (str): Chromosome name (e.g., ``"chr1"``).
            start_pos (int): Genomic start position (1-based index).

        Returns:
            dict: A JSON response containing the predicted coverage track.
        """
        return self.rna.predict(chrom, start_pos)

