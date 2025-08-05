"""核心数据模型和Schema定义"""
from .base import (
    StrictBaseModel,
    PageSpan,
    BBox,
    FigureType,
    VariableRole,
    AxisScale
)

from .document import (
    Section,
    Affiliation,
    Author,
    Reference,
    DocumentAnnotation
)

from .bbox import (
    Variable,
    Axis,
    BBoxAnnotation,
    BBoxPage
)

from .json_schemas import (
    generate_json_schema,
    save_schemas_to_config,
    get_document_schema_for_mistral,
    get_bbox_schema_for_mistral
)

__all__ = [
    # Base
    'StrictBaseModel',
    'PageSpan',
    'BBox',
    'FigureType',
    'VariableRole', 
    'AxisScale',
    
    # Document
    'Section',
    'Affiliation',
    'Author',
    'Reference',
    'DocumentAnnotation',
    
    # BBox
    'Variable',
    'Axis',
    'BBoxAnnotation',
    'BBoxPage',
    
    # Schema utilities
    'generate_json_schema',
    'save_schemas_to_config',
    'get_document_schema_for_mistral',
    'get_bbox_schema_for_mistral'
]