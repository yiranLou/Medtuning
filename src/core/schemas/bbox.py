from typing import List, Optional, Tuple
from pydantic import Field, field_validator, model_validator
import re
from pathlib import Path
from .base import StrictBaseModel, BBox, FigureType, VariableRole, AxisScale


class Variable(StrictBaseModel):
    """图表变量"""
    name: str = Field(..., min_length=1, max_length=100, description="变量名")
    role: VariableRole = Field(..., description="变量角色")
    unit: Optional[str] = Field(None, max_length=50, description="单位，标准化为SI单位或空串")
    category_values: Optional[List[str]] = Field(None, description="分类变量的枚举值")
    
    @field_validator('unit')
    def standardize_unit(cls, v):
        if not v:
            return None
        
        # 标准化常见单位
        unit_map = {
            'ml': 'mL', 'ML': 'mL',
            'mg': 'mg', 'MG': 'mg',
            'ug': 'μg', 'µg': 'μg', 'mcg': 'μg',
            'ng': 'ng', 'NG': 'ng',
            'pg': 'pg', 'PG': 'pg',
            'kg': 'kg', 'KG': 'kg',
            'g': 'g', 'G': 'g',
            'l': 'L', 'L': 'L',
            'dl': 'dL', 'DL': 'dL',
            'ul': 'μL', 'µl': 'μL', 'uL': 'μL',
            'mol': 'mol', 'MOL': 'mol',
            'mmol': 'mmol', 'MMOL': 'mmol',
            'umol': 'μmol', 'µmol': 'μmol',
            'nm': 'nm', 'NM': 'nm',
            'um': 'μm', 'µm': 'μm',
            'mm': 'mm', 'MM': 'mm',
            'cm': 'cm', 'CM': 'cm',
            'm': 'm', 'M': 'm',
            'h': 'h', 'hr': 'h', 'hour': 'h',
            'min': 'min', 'minute': 'min',
            's': 's', 'sec': 's', 'second': 's',
            'day': 'd', 'days': 'd',
            'week': 'week', 'weeks': 'week',
            'month': 'month', 'months': 'month',
            'year': 'year', 'years': 'year',
            '°c': '°C', '℃': '°C', 'celsius': '°C',
            '°f': '°F', '℉': '°F', 'fahrenheit': '°F',
            'k': 'K', 'kelvin': 'K',
            '%': '%', 'percent': '%',
            'pa': 'Pa', 'PA': 'Pa',
            'kpa': 'kPa', 'KPA': 'kPa',
            'mmhg': 'mmHg', 'MMHG': 'mmHg',
        }
        
        v = v.strip()
        return unit_map.get(v.lower(), v)
    
    @model_validator(mode='after')
    def validate_category_values(self):
        """分类变量才能有枚举值"""
        if self.role not in [VariableRole.GROUP, VariableRole.SERIES, VariableRole.LEGEND]:
            if self.category_values:
                raise ValueError(f"角色{self.role}不应有category_values")
        return self


class Axis(StrictBaseModel):
    """坐标轴信息"""
    x_label: Optional[str] = Field(None, max_length=200, description="X轴标签")
    y_label: Optional[str] = Field(None, max_length=200, description="Y轴标签")
    x_unit: Optional[str] = Field(None, max_length=50, description="X轴单位")
    y_unit: Optional[str] = Field(None, max_length=50, description="Y轴单位")
    scale: Optional[AxisScale] = Field(None, description="刻度类型")
    
    @field_validator('x_unit', 'y_unit')
    def standardize_units(cls, v):
        if not v:
            return None
        # 复用Variable的单位标准化逻辑
        return Variable.model_validate({'name': 'temp', 'role': VariableRole.X, 'unit': v}).unit


class BBoxAnnotation(StrictBaseModel):
    """边界框级标注"""
    paper_id: str = Field(..., min_length=1, max_length=100, description="论文唯一ID")
    page_index: int = Field(..., ge=0, description="页码索引，从0开始")
    bbox: BBox = Field(..., description="边界框坐标")
    crop_path: str = Field(..., min_length=1, description="裁剪图相对路径")
    figure_type: FigureType = Field(..., description="图表类型")
    caption: Optional[str] = Field(None, max_length=1000, description="图表标题，去除编号")
    variables: Optional[List[Variable]] = Field(None, description="变量列表")
    axis: Optional[Axis] = Field(None, description="坐标轴信息")
    key_findings: Optional[str] = Field(None, max_length=200, description="关键发现，一句话总结")
    table_csv: Optional[str] = Field(None, max_length=10000, description="表格CSV内容")
    table_path: Optional[str] = Field(None, description="外部表格文件路径")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="标注置信度")
    
    @field_validator('paper_id')
    def validate_paper_id(cls, v):
        if not re.match(r'^(PMC\d+|arXiv:\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z0-9_-]+)$', v):
            raise ValueError("paper_id格式不正确")
        return v
    
    @field_validator('crop_path')
    def validate_crop_path(cls, v):
        # 确保是相对路径
        if Path(v).is_absolute():
            raise ValueError("crop_path必须是相对路径")
        # 确保是图片格式
        if not v.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("crop_path必须是图片文件")
        return v
    
    @field_validator('caption')
    def clean_caption(cls, v):
        if v:
            # 去除图表编号前缀
            v = re.sub(r'^(Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?)\s*\d+[:\.]?\s*', '', v, flags=re.IGNORECASE)
            # 去除控制字符
            v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
            return v.strip()
        return v
    
    @field_validator('key_findings')
    def validate_key_findings(cls, v):
        if v:
            # 去除控制字符
            v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
            # 检查长度（中文按2字符计算）
            char_count = len(v) + len(re.findall(r'[\u4e00-\u9fa5]', v))
            if char_count > 100:  # 50个中文字符
                raise ValueError("key_findings过长，应<=50个中文字")
            # 禁止推断性词汇
            forbidden_patterns = [
                r'可能', r'也许', r'或许', r'大概', r'推测', r'猜测',
                r'might', r'maybe', r'perhaps', r'probably', r'possibly',
                r'could be', r'may be', r'seems', r'appears'
            ]
            for pattern in forbidden_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError(f"key_findings不应包含推断性词汇: {pattern}")
            return v
        return v
    
    @model_validator(mode='after')
    def validate_table_consistency(self):
        """表格类型一致性验证"""
        if self.figure_type == FigureType.TABLE:
            if not self.table_csv and not self.table_path:
                # 表格类型应该有CSV或路径
                pass  # 允许为空，但记录warning
        else:
            if self.table_csv or self.table_path:
                raise ValueError("非表格类型不应有table_csv或table_path")
        return self
    
    @model_validator(mode='after')
    def validate_axis_consistency(self):
        """坐标轴信息一致性验证"""
        if self.figure_type in [FigureType.FIGURE, FigureType.DIAGRAM]:
            # 图表类型应该有坐标轴信息
            pass
        elif self.axis and (self.axis.x_label or self.axis.y_label):
            # 非图表类型不应有坐标轴
            if self.figure_type not in [FigureType.FLOWCHART, FigureType.OTHER]:
                raise ValueError(f"{self.figure_type}类型不应有坐标轴信息")
        return self
    
    def validate_bbox_within_page(self, page_width: int, page_height: int) -> None:
        """验证边界框在页面范围内"""
        if self.bbox.x2 > page_width:
            raise ValueError(f"bbox.x2({self.bbox.x2})超出页面宽度({page_width})")
        if self.bbox.y2 > page_height:
            raise ValueError(f"bbox.y2({self.bbox.y2})超出页面高度({page_height})")


class BBoxPage(StrictBaseModel):
    """单页的所有边界框标注"""
    paper_id: str
    page_index: int
    page_width: int = Field(..., gt=0, description="页面宽度（像素）")
    page_height: int = Field(..., gt=0, description="页面高度（像素）")
    annotations: List[BBoxAnnotation] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_annotations(self):
        """验证所有标注的一致性"""
        for ann in self.annotations:
            # 验证paper_id一致
            if ann.paper_id != self.paper_id:
                raise ValueError("标注的paper_id与页面不一致")
            # 验证page_index一致
            if ann.page_index != self.page_index:
                raise ValueError("标注的page_index与页面不一致")
            # 验证bbox在页面范围内
            ann.validate_bbox_within_page(self.page_width, self.page_height)
        return self