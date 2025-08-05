# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-05

### Added
- Initial release of MedTuning
- Two-layer schema design for document and bbox annotations
- PDF processing with PyMuPDF
- Mistral Document AI integration
- InternVL2 JSONL dataset generation
- Quality control and validation system
- Configurable Q/A templates
- Support for medical terminology and units
- Batch processing capabilities
- Comprehensive documentation
- Example configurations and templates

### Features
- Document-level annotation extraction (title, abstract, sections, authors)
- Bounding box annotation for figures and tables
- Multiple task types support:
  - Page grounding (element localization)
  - Figure caption generation
  - Variable extraction
  - Table reading
  - Multi-figure comparison
  - Abstract Q&A
- Automatic deduplication
- Schema validation
- Coordinate verification
- Sampling strategies

### Technical
- Python 3.8+ support
- Async/await for API calls
- Pydantic for data validation
- Configurable rendering DPI
- Memory-efficient batch processing

## [Unreleased]

### Planned
- Local OCR model support
- More figure detection backends
- Enhanced table extraction
- Multi-language support
- Web UI for visualization
- Docker deployment