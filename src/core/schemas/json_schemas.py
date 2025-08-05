"""JSON Schema生成器，用于Mistral API调用"""
import json
from pathlib import Path
from typing import Dict, Any

from .document import DocumentAnnotation
from .bbox import BBoxAnnotation, BBoxPage


def generate_json_schema(model_class) -> Dict[str, Any]:
    """从Pydantic模型生成JSON Schema"""
    schema = model_class.model_json_schema()
    
    # 添加额外的约束
    schema['additionalProperties'] = False
    
    # 递归处理所有定义
    if '$defs' in schema:
        for def_name, def_schema in schema['$defs'].items():
            def_schema['additionalProperties'] = False
    
    return schema


def save_schemas_to_config(output_dir: Path):
    """保存所有Schema到配置目录"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schemas = {
        'document_annotation': generate_json_schema(DocumentAnnotation),
        'bbox_annotation': generate_json_schema(BBoxAnnotation),
        'bbox_page': generate_json_schema(BBoxPage)
    }
    
    for name, schema in schemas.items():
        output_file = output_dir / f'{name}_schema.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"已保存 {name} schema 到 {output_file}")
    
    return schemas


def get_document_schema_for_mistral() -> Dict[str, Any]:
    """获取用于Mistral API的文档Schema（简化版）"""
    schema = generate_json_schema(DocumentAnnotation)
    
    # Mistral可能需要的额外配置
    schema['description'] = "学术论文文档级结构化标注"
    
    # 移除一些Mistral可能不支持的特性
    if '$defs' in schema:
        # 将引用展开
        schema = _expand_refs(schema)
    
    return schema


def get_bbox_schema_for_mistral() -> Dict[str, Any]:
    """获取用于Mistral API的BBox Schema（简化版）"""
    schema = generate_json_schema(BBoxAnnotation)
    
    schema['description'] = "学术论文图表级边界框标注"
    
    # 展开引用
    if '$defs' in schema:
        schema = _expand_refs(schema)
    
    return schema


def _expand_refs(schema: Dict[str, Any], defs: Dict[str, Any] = None) -> Dict[str, Any]:
    """展开JSON Schema中的$ref引用"""
    if defs is None and '$defs' in schema:
        defs = schema['$defs']
        schema = schema.copy()
        del schema['$defs']
    
    if isinstance(schema, dict):
        if '$ref' in schema and defs:
            ref_path = schema['$ref'].split('/')[-1]
            if ref_path in defs:
                return _expand_refs(defs[ref_path], defs)
        else:
            return {k: _expand_refs(v, defs) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [_expand_refs(item, defs) for item in schema]
    else:
        return schema


if __name__ == '__main__':
    # 测试生成Schema
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    # 保存到配置目录
    config_dir = Path(__file__).parent.parent.parent.parent / 'configs' / 'schemas'
    save_schemas_to_config(config_dir)