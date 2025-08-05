"""PDF!W"""
from .renderer import PDFRenderer, RenderConfig
from .detector import (
    FigureDetector,
    DetectedFigure,
    PDFFigures2Detector,
    HeuristicDetector,
    CombinedDetector,
    create_detector
)

__all__ = [
    'PDFRenderer',
    'RenderConfig',
    'FigureDetector',
    'DetectedFigure',
    'PDFFigures2Detector',
    'HeuristicDetector',
    'CombinedDetector',
    'create_detector'
]