Release Notes
=============

Version: v1.0.2
---------------
**Release Date:** Oct 30, 2026

### Changes
- Package version is 1.0.2. Variant requests send this version to the scoring service.
- Variant prediction calls the DCS endpoint and defaults to the deployed model `Genos_Variant`.
- Only `hg38` coordinates are accepted. The default assembly is `hg38`.
- `result["score"]` is the published pathogenic score (`p_fusion` when the service returns it).

Version: v0.2.0
---------------
**Release Date:** Oct 29, 2026

### New Features
- Variant prediction now requests the E85 published scores.
- `result["score"]` is the published fusion score (`p_fusion`). `p_mlp` and `p_evee` are included alongside it.
- Each variant request sends `client_version` 0.2.0. A rejected version raises `PackageUpdateRequired`.

### Compatibility
- Calls must use `hg38` coordinates.
- Installed 0.1.0 packages are outside the server allowlist. They receive an upgrade notice in `result` and `message`.

Version: v0.1.0
---------------
**Release Date:** Oct 10, 2025

### New Features
- Initial release of **Genos SDK**.
- Added support for:
  - `RNASeqCoverageTrackPredictionAPI` for RNA-seq coverage track prediction.
  - `VariantPredictor` for variant effect prediction.
  - `EmbeddingExtractor` for feature extraction from biological sequences.
- Implemented unified `GenosClient` interface for managing API access and authentication.

### Improvements
- Improved modular design for API endpoints.
- Added type hints and docstrings for better IDE integration and documentation generation.


### Documentation
- Added installation guide and quick start tutorial.
- Integrated API documentation with `sphinx.ext.autodoc` and `sphinx.ext.viewcode`.
