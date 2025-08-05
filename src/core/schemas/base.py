from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re
from enum import Enum


class StrictBaseModel(BaseModel):
    """基础模型，启用严格模式和额外字段禁止"""
    model_config = ConfigDict(
        extra='forbid',  # 禁止额外字段
        str_strip_whitespace=True,  # 自动去除字符串两端空白
        validate_assignment=True,  # 赋值时验证
        use_enum_values=True
    )


class PageSpan(BaseModel):
    """页面范围"""
    page: int = Field(..., ge=0, description="页码，从0开始")
    y1: float = Field(..., ge=0, description="起始Y坐标")
    y2: float = Field(..., gt=0, description="结束Y坐标")
    
    @field_validator('y2')
    def validate_y2(cls, v, info):
        if 'y1' in info.data and v <= info.data['y1']:
            raise ValueError('y2必须大于y1')
        return v


class BBox(BaseModel):
    """边界框坐标"""
    x1: int = Field(..., ge=0, description="左上角X坐标")
    y1: int = Field(..., ge=0, description="左上角Y坐标")
    x2: int = Field(..., gt=0, description="右下角X坐标")
    y2: int = Field(..., gt=0, description="右下角Y坐标")
    
    @field_validator('x2')
    def validate_x2(cls, v, info):
        if 'x1' in info.data and v <= info.data['x1']:
            raise ValueError('x2必须大于x1')
        return v
    
    @field_validator('y2')
    def validate_y2(cls, v, info):
        if 'y1' in info.data and v <= info.data['y1']:
            raise ValueError('y2必须大于y1')
        return v
    
    def to_list(self) -> List[int]:
        """转换为列表格式[x1,y1,x2,y2]"""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, bbox_list: List[int]) -> 'BBox':
        """从列表创建BBox"""
        if len(bbox_list) != 4:
            raise ValueError("BBox列表必须包含4个元素")
        return cls(x1=bbox_list[0], y1=bbox_list[1], x2=bbox_list[2], y2=bbox_list[3])


class FigureType(str, Enum):
    """图表类型枚举"""
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    OTHER = "other"


class VariableRole(str, Enum):
    """变量角色枚举"""
    X = "x"
    Y = "y"
    GROUP = "group"
    SERIES = "series"
    CONFIDENCE_INTERVAL = "confidence_interval"
    LEGEND = "legend"


class AxisScale(str, Enum):
    """坐标轴刻度类型"""
    LINEAR = "linear"
    LOG = "log"