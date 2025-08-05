from typing import List, Optional
from pydantic import Field, field_validator, model_validator
import re
from .base import StrictBaseModel, PageSpan


class Section(StrictBaseModel):
    """文档章节"""
    title: str = Field(..., min_length=1, max_length=500, description="章节标题")
    level: int = Field(..., ge=1, le=6, description="章节级别，1-6")
    text: str = Field(..., min_length=1, max_length=32000, description="章节内容，合并同级段落")
    page_spans: Optional[List[PageSpan]] = Field(None, description="页面范围")
    
    @field_validator('title')
    def validate_title(cls, v):
        # 去除尾随标点
        v = re.sub(r'[。，；：！？\.,:;!?]+$', '', v)
        # 去除控制字符
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        return v
    
    @field_validator('text')
    def validate_text(cls, v):
        # 去除控制字符
        v = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', v)
        return v


class Affiliation(StrictBaseModel):
    """作者单位"""
    name: str = Field(..., min_length=1, max_length=500, description="单位名称")
    department: Optional[str] = Field(None, max_length=300, description="部门")
    city: Optional[str] = Field(None, max_length=100, description="城市")
    country: Optional[str] = Field(None, max_length=100, description="国家")
    
    @field_validator('name', 'department', 'city', 'country')
    def clean_text(cls, v):
        if v:
            return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        return v


class Author(StrictBaseModel):
    """作者信息"""
    name: str = Field(..., min_length=1, max_length=200, description="作者姓名")
    email: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    orcid: Optional[str] = Field(None, pattern=r'^\d{4}-\d{4}-\d{4}-\d{3}[0-9X]$')
    affiliation_ids: List[int] = Field(default_factory=list, description="单位ID列表")
    
    @field_validator('name')
    def clean_name(cls, v):
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)


class Reference(StrictBaseModel):
    """参考文献"""
    raw_text: str = Field(..., min_length=1, max_length=2000, description="原始引用文本")
    doi: Optional[str] = Field(None, pattern=r'^10\.\d{4,9}/[-._;()\/:a-zA-Z0-9]+$')
    pmid: Optional[str] = Field(None, pattern=r'^\d+$')
    
    @field_validator('raw_text')
    def clean_reference(cls, v):
        # 基本清洗：去除控制字符，保留换行
        v = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', v)
        # 去除多余空白
        v = re.sub(r'\s+', ' ', v).strip()
        return v


class DocumentAnnotation(StrictBaseModel):
    """文档级标注"""
    paper_id: str = Field(..., min_length=1, max_length=100, description="论文唯一ID")
    title: str = Field(..., min_length=1, max_length=500, description="论文标题")
    abstract: str = Field(..., min_length=1, max_length=5000, description="摘要")
    keywords: Optional[List[str]] = Field(None, max_length=10, description="关键词列表")
    sections: List[Section] = Field(..., min_length=1, description="章节列表")
    authors: Optional[List[Author]] = Field(None, description="作者列表")
    affiliations: Optional[List[Affiliation]] = Field(None, description="单位列表")
    references: Optional[List[Reference]] = Field(None, description="参考文献列表")
    doi: Optional[str] = Field(None, pattern=r'^10\.\d{4,9}/[-._;()\/:a-zA-Z0-9]+$')
    journal: Optional[str] = Field(None, max_length=300, description="期刊名称")
    publication_date: Optional[str] = Field(None, pattern=r'^\d{4}(-\d{2}(-\d{2})?)?$')
    
    @field_validator('paper_id')
    def validate_paper_id(cls, v):
        # 支持PMC、arXiv等格式
        if not re.match(r'^(PMC\d+|arXiv:\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z0-9_-]+)$', v):
            raise ValueError("paper_id格式不正确")
        return v
    
    @field_validator('title', 'abstract')
    def clean_text_fields(cls, v):
        # 去除控制字符
        v = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', v)
        # 去除尾随标点（仅对title）
        if cls.__name__ == 'title':
            v = re.sub(r'[。，；：！？\.,:;!?]+$', '', v)
        return v
    
    @field_validator('keywords')
    def validate_keywords(cls, v):
        if v:
            # 转小写、去重、清洗
            cleaned = []
            seen = set()
            for kw in v:
                kw = kw.lower().strip()
                kw = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', kw)
                if kw and kw not in seen:
                    cleaned.append(kw)
                    seen.add(kw)
            return cleaned[:10]  # 最多10个
        return v
    
    @model_validator(mode='after')
    def validate_empty_arrays(self):
        """移除空数组字段"""
        for field_name in ['keywords', 'authors', 'affiliations', 'references']:
            value = getattr(self, field_name)
            if value is not None and len(value) == 0:
                setattr(self, field_name, None)
        return self
    
    @model_validator(mode='after')
    def validate_author_affiliations(self):
        """验证作者单位ID的有效性"""
        if self.authors and self.affiliations:
            valid_ids = set(range(len(self.affiliations)))
            for author in self.authors:
                for aff_id in author.affiliation_ids:
                    if aff_id not in valid_ids:
                        raise ValueError(f"作者{author.name}的单位ID {aff_id} 无效")
        return self