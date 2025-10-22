![Genos header image](docs/_static/geno.png)

# Genos

[**Get API key**](#get-api-key) |
[**Quick start**](#quick-start) 

**Genos** is a Python SDK for accessing genomic analysis models through the GeneOS platform. It provides a unified interface for variant pathogenicity prediction, DNA sequence embedding extraction, RNA-seq coverage track prediction, and genomic visualization.


## Features

- 🧬 **Variant Prediction**: Assess pathogenicity of genetic variants
- 🔬 **Embedding Extraction**: Extract deep learning embeddings from DNA sequences
- 🧪 **RNA-seq Coverage Track Prediction**: Predict RNA-seq coverage tracks from genomic coordinates
- 📊 **Genomic Visualization**: Plot and analyze genomic tracks
- 🚀 **Easy to Use**: Simple, intuitive API with comprehensive error handling
- 🔐 **Robust Authentication**: Automatic token validation and payment checking
- ⚠️ **Comprehensive Error Handling**: Specific exceptions for different error types



## Get API Key

To use Genos, you need an API key from the DCS Cloud.

### Request Access
1. Log in to the [DCS Clould](https://cloud.stomics.tech/#/login) and navigate to **Personal Center → API Key Management**.
2. Click **“Create API Key”**.
3. Read the **“API Usage Notice”** and confirm your agreement.
4. The system will automatically generate your exclusive API Key — **copy and keep it safe**.
> ⚠️ Please store your key securely and avoid any unauthorized disclosure.

### Usage Policy
* The API Key is **for personal use only** — sharing, transferring, or publishing is strictly prohibited.
* If the key is **leaked, misused, or used for illegal purposes**, the platform reserves the right to **immediately disable it**.
* You can **manually deactivate** your key in the control panel at any time.

## Installation
### Install from Source
```bash
git clone https://github.com/BGI-HangzhouAI/Genos.git
cd sdk
pip install -e .
```

### Install from Pypi
```bash
pip install genos
```

### Requirements

- Python 3.8 or higher
- pip package manager



## Quick Start

### Basic Usage

```python
from genos import create_client

# Create client (uses GENOS_API_TOKEN environment variable)
client = create_client()

# Or provide token explicitly
client = create_client(token="your_api_token_here")
```

### 1. Variant Pathogenicity Prediction

Predict whether a genetic variant is pathogenic or benign:

```python
# Predict variant pathogenicity
result = client.variant_predict("hg19", "chr6", 51484075, "T", "G")['result']

print(f"Variant: {result['variant']}")
print(f"Prediction: {result['prediction']}")
print(f"Pathogenic Score: {result['score_Pathogenic']:.4f}")
print(f"Benign Score: {result['score_Benign']:.4f}")
```

### 2. DNA Sequence Embedding Extraction

Extract deep learning embeddings from DNA sequences:

```python
# Extract embedding for a single sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCG"
result = client.get_embedding(sequence, model_name="Genos-1.2B")['result']

print(f"Sequence Length: {result['sequence_length']}")
print(f"Embedding Dimension: {result['embedding_dim']}")
print(f"Embedding Shape: {result['embedding_shape']}")

# Access the embedding vector
embedding_vector = result['embedding']  # List of floats
```

**Available Models:**
- `Genos-1.2B`: 1.2 billion parameter model
- `Genos-10B`: 10 billion parameter model

**Pooling Methods:**
- `mean`: Average pooling across sequence
- `max`: Max pooling
- `last`: Use last token embedding
- `none`: Return all token embeddings

### 3. RNA-seq Coverage Track Prediction

Predict RNA-seq coverage tracks based on genomic coordinates:

```python
# Predict RNA-seq coverage track
result = client.rna_coverage_track_pred(chrom="chr6", start_pos=51484075)['result']

print(f"Predicted coverage track: {result}")
```


## Advanced Configuration

### Custom Embedding Service

GenosClient allows users to configure a **custom embedding API endpoint**.  
This is useful if you want to deploy your own embedding service locally or within your organization.  

**Note:** Now, the variant prediction and RNA-seq coverage track prediction models are **not open-source** and cannot be self-hosted. Only the embedding service can be customized.

```python
from genos import GenosClient

# Initialize the client with a custom embedding endpoint
client = GenosClient(
    token="your_custom_token",  # Your token for authenticating with your own embedding service
    api_map={
        # Only the embedding service can be customized
        "embedding": "https://custom-embed-api.example.com/predict"
    }
)

# Calls to variant and RNA APIs will still use the official hosted services
```

### Timeout Configuration

Adjust request timeout for long-running operations:

```python
# Set 60-second timeout
client = create_client(token="your_token", timeout=60)
```

## Error Handling

Genos provides comprehensive error handling with specific exception types for different scenarios. All API responses follow a consistent format.

### Error Response Format

All error responses from the Genos API follow this structure:

```json
{
  "result": {},
  "status": "<HTTP_STATUS_CODE>",
  "messages": "<ERROR_MESSAGE>"
}
```

### Common Error Codes

| Status Code | Error Message | Description |
|-------------|---------------|-------------|
| 400 | Insufficient balance | Your account balance is insufficient for the requested operation |
| 401 | Invalid API Key | The provided API key is invalid or expired |
| 500 | Internal server error | An unexpected error occurred on the server side |

## Examples

Complete examples are available in the [`examples/`](examples/) directory:

- [`predict_variant.py`](examples/predict_variant.py): Variant pathogenicity prediction
- [`embedding_extract.py`](examples/embedding_extract.py): DNA Sequence Embedding Extraction
- [`rna_generator.py`](examples/rna_generator.py): RNA-seq Coverage Track Prediction
- [`error_handling_demo.py`](examples/error_handling_demo.py): Comprehensive error handling examples



## Primary Paper（**Replace with real information**）

xxx, xxx, *et al.*  
*Genos: A Unified Foundation Model for Genomic Sequence Understanding and Variant Interpretation.（ju）*  
_GigaScience_, 2025.  
[https://doi.org/10.xxxx/genos.2025.12345](https://doi.org/10.xxxx/genos.2025.12345)


## How to Cite （**Replace with real information**）

If you use the SDK, please include the following citation in your work:

```bibtex
@article{he2025genos,
  title   = {Genos: A Unified Foundation Model for Genomic Sequence Understanding and Variant Interpretation},
  author  = {xxx},
  journal = {GigaSience},
  year    = {2025},
  doi     = {10.xxxx/genos.2025.12345}
}
