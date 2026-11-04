#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
import time
import logging
from .health import BaseAPI
from ..exceptions import APIRequestError, PackageUpdateRequired
from ..version import __version__

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


def _json_container(value):
    if isinstance(value, str):
        text = value.strip()
        if not text or text[0] not in "{[":
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return value


def _iter_mappings(value):
    value = _json_container(value)
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_mappings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_mappings(item)


def _error_text(data, status_code) -> str:
    """Read a gateway error from messages, message, or error.message."""
    if not isinstance(data, dict):
        return f"HTTP {status_code} error"
    error = data.get("error")
    if isinstance(error, dict):
        nested = error.get("message") or error.get("messages")
        if nested:
            return str(nested)
    if isinstance(error, str) and error:
        return error
    for key in ("messages", "message"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return f"HTTP {status_code} error"


def _response_preview(data) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, default=str)
    except TypeError:
        text = str(data)
    if len(text) > 500:
        return text[:500] + "..."
    return text


def _upgrade_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    markers = ("不在允许列表", "请将 genos", "Please update the genos package", "update_required")
    if any(marker in value for marker in markers):
        return value
    return None


def update_required_message(data: dict):
    """Return the server upgrade notice, or None when the call may proceed."""
    if not isinstance(data, dict):
        return None
    if data.get("update_required"):
        message = data.get("message") or data.get("messages") or data.get("result")
        if isinstance(message, dict):
            message = message.get("message") or message.get("prediction")
        return message or "请更新 genos 包后再调用变异预测。"
    for mapping in _iter_mappings(data):
        if mapping.get("update_required"):
            return (
                mapping.get("message")
                or mapping.get("prediction")
                or data.get("message")
                or data.get("messages")
                or "请更新 genos 包后再调用变异预测。"
            )
        for key in ("message", "messages", "prediction", "variant"):
            notice = _upgrade_text(mapping.get(key))
            if notice:
                return notice
    return None


def _as_score(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _score_source(data: dict):
    """Prefer the new published fusion score, then the gateway's pathogenic score."""
    mappings = list(_iter_mappings(data))
    for mapping in mappings:
        if _as_score(mapping.get("p_fusion")) is not None:
            return mapping
    for mapping in mappings:
        if _as_score(mapping.get("score_Pathogenic")) is not None:
            return mapping
    return None


def published_variant_result(data: dict) -> dict:
    """Read a published score from either the new or the gateway response.

    The new service returns ``p_fusion``. The Stomics gateway currently returns
    ``result.score_Pathogenic`` and ``result.score_Benign`` inside
    ``{result, status, messages}``. ``score`` is the pathogenic probability
    from whichever of those is present.
    """
    source = _score_source(data)
    if source is None:
        raise APIRequestError(
            "响应里没有 published score（p_fusion 或 score_Pathogenic）。"
            f"实际返回：{_response_preview(data)}"
        )
    p_fusion = _as_score(source.get("p_fusion"))
    score_pathogenic = _as_score(source.get("score_Pathogenic"))
    if p_fusion is None:
        p_fusion = score_pathogenic
    if score_pathogenic is None:
        score_pathogenic = p_fusion
    return {
        "variant": source.get("variant"),
        "chromosome": source.get("chromosome"),
        "position": source.get("position"),
        "ref": source.get("ref"),
        "alt": source.get("alt"),
        "prediction": source.get("prediction"),
        "p_mlp": _as_score(source.get("p_mlp")),
        "p_evee": _as_score(source.get("p_evee")),
        "p_fusion": p_fusion,
        "score": p_fusion,
        "score_Pathogenic": score_pathogenic,
        "score_Benign": _as_score(source.get("score_Benign")),
        "fusion_threshold": _as_score(source.get("fusion_threshold")),
        "elapsed_ms": source.get("elapsed_ms"),
    }


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

    def __init__(self, session: requests.Session, base_url: str, timeout: int = 30, model: str = "Genos_Variant"):
        """
        Initialize the VariantPredictorAPI client.

        Args:
            session (requests.Session): Reusable HTTP session for API requests.
            base_url (str): Base URL of the Genos service.
            timeout (int, optional): Request timeout in seconds. Default is 30.
        """
        super().__init__(session, base_url, timeout)
        self.model = model

    def predict(self, assembly: str = "hg38", chrom: str = None, pos: int = None, ref: str = None, alt: str = None, model: str = None) -> dict:
        """
        Predict the pathogenicity of a genetic variant.
        
        Args:
            assembly (str): Reference genome. Only ``'hg38'`` is accepted. Default: ``'hg38'``.
            chrom (str): Chromosome, e.g., 'chr6'.
            pos (int): Position, 1-based coordinate.
            ref (str): Reference allele, single letter or sequence.
            alt (str): Alternate allele, single letter or sequence.

        Returns:
            dict: ``{"result", "status", "message"}``. ``result["score"]`` is the
            published fusion score (``p_fusion``). ``result`` also includes
            ``p_mlp`` and ``p_evee``.

        Raises:
            ValueError: If any argument is invalid.
            PackageUpdateRequired: If this package version is not allowed.
            APIRequestError: If the API request fails or the server returns an error.
        """
        if assembly is None:
            assembly = "hg38"
        if str(assembly).strip().lower() != "hg38":
            raise ValueError(f"assembly must be 'hg38', but got '{assembly}'")
        assembly = "hg38"

        if chrom is None or pos is None or ref is None or alt is None:
            raise ValueError("chrom, pos, ref and alt are required")

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
            "model": model or self.model,
            "assembly": assembly,
            "chrom": chrom,
            "pos": pos,
            "ref": ref.upper(),
            "alt": alt.upper(),
            "client_version": __version__,
        }

        start_time = time.time()
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise APIRequestError(f"Network request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise APIRequestError(f"API response is not valid JSON: {exc}") from exc
        notice = update_required_message(data)
        if notice:
            raise PackageUpdateRequired(str(notice), __version__)

        body_status = data.get("status") if isinstance(data, dict) else None
        if response.status_code == 200 and body_status not in (None, 200, "200"):
            error_message = _error_text(data, body_status)
            raise APIRequestError(f"API request failed: {error_message}", body_status)

        if response.status_code != 200:
            error_message = _error_text(data, response.status_code)
            print(f"❌ API Error - Status Code: {response.status_code}")
            print(f"❌ Error Message: {error_message}")
            raise APIRequestError(f"API request failed: {error_message}", response.status_code)

        result = published_variant_result(data)
        elapsed_time = time.time() - start_time
        logger.info(f"Variant prediction completed in {elapsed_time:.4f}s ")
        return {
            "result": result,
            "status": data.get("status", 200),
            "message": data.get("message") or data.get("messages") or "ok",
        }
