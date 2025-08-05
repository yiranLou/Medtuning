"""数据一致性检查器"""
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from collections import defaultdict

from ..core.schemas import (
    DocumentAnnotation,
    BBoxAnnotation,
    BBox,
    FigureType
)
from ..dataset import InternVL2Sample

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """数据一致性检查器"""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
    
    def check_document_annotation(self, annotation: DocumentAnnotation) -> bool:
        """检查文档标注一致性"""
        self.errors = []
        self.warnings = []
        
        # 1. 检查paper_id格式
        if not self._validate_paper_id(annotation.paper_id):
            self.errors.append(f"无效的paper_id格式: {annotation.paper_id}")
        
        # 2. 检查标题和摘要
        if not annotation.title or len(annotation.title.strip()) < 5:
            self.errors.append("标题过短或为空")
        
        if not annotation.abstract or len(annotation.abstract.strip()) < 50:
            self.errors.append("摘要过短或为空")
        
        # 3. 检查章节
        if not annotation.sections:
            self.errors.append("缺少章节信息")
        else:
            self._check_sections_consistency(annotation.sections)
        
        # 4. 检查作者和单位
        if annotation.authors and annotation.affiliations:
            self._check_author_affiliations(annotation.authors, annotation.affiliations)
        
        # 5. 检查DOI格式
        if annotation.doi and not self._validate_doi(annotation.doi):
            self.warnings.append(f"DOI格式可能不正确: {annotation.doi}")
        
        # 6. 检查日期格式
        if annotation.publication_date and not self._validate_date(annotation.publication_date):
            self.errors.append(f"日期格式错误: {annotation.publication_date}")
        
        return len(self.errors) == 0 if self.strict_mode else len(self.errors) < 3
    
    def check_bbox_annotation(
        self, 
        annotation: BBoxAnnotation,
        page_width: int,
        page_height: int
    ) -> bool:
        """检查边界框标注一致性"""
        self.errors = []
        self.warnings = []
        
        # 1. 检查坐标范围
        if not self._validate_bbox_coords(annotation.bbox, page_width, page_height):
            self.errors.append(
                f"边界框坐标超出页面范围: {annotation.bbox.to_list()} "
                f"页面尺寸: {page_width}x{page_height}"
            )
        
        # 2. 检查图表类型一致性
        self._check_figure_type_consistency(annotation)
        
        # 3. 检查裁剪路径
        if not annotation.crop_path or not self._validate_image_path(annotation.crop_path):
            self.errors.append(f"无效的裁剪图路径: {annotation.crop_path}")
        
        # 4. 检查变量和轴的一致性
        if annotation.variables and annotation.axis:
            self._check_variable_axis_consistency(annotation.variables, annotation.axis)
        
        # 5. 检查key_findings
        if annotation.key_findings:
            self._check_key_findings(annotation.key_findings)
        
        # 6. 检查表格数据
        if annotation.figure_type == FigureType.TABLE:
            if not annotation.table_csv and not annotation.table_path:
                self.warnings.append("表格类型但缺少table_csv或table_path")
        else:
            if annotation.table_csv or annotation.table_path:
                self.errors.append("非表格类型不应有table_csv或table_path")
        
        return len(self.errors) == 0 if self.strict_mode else len(self.errors) < 2
    
    def check_internvl2_sample(self, sample: InternVL2Sample) -> bool:
        """检查InternVL2样本一致性"""
        self.errors = []
        self.warnings = []
        
        # 1. 检查图片数量与<image>标记
        conversations_text = " ".join([c["value"] for c in sample.conversations])
        image_tag_count = conversations_text.count("<image>")
        
        if isinstance(sample.image, str):
            expected_count = 1
        else:
            expected_count = len(sample.image)
        
        if image_tag_count != expected_count:
            self.errors.append(
                f"<image>标记数量({image_tag_count})与图片数量({expected_count})不匹配"
            )
        
        # 2. 检查宽高数据
        if isinstance(sample.image, list):
            if not isinstance(sample.width, list) or len(sample.width) != len(sample.image):
                self.errors.append("多图样本的width_list长度不匹配")
            if not isinstance(sample.height, list) or len(sample.height) != len(sample.image):
                self.errors.append("多图样本的height_list长度不匹配")
        
        # 3. 检查对话格式
        if not self._validate_conversations(sample.conversations):
            self.errors.append("对话格式错误")
        
        # 4. 检查grounding坐标
        if "<box>" in conversations_text:
            self._check_grounding_coords(conversations_text, sample.width, sample.height)
        
        return len(self.errors) == 0
    
    def _validate_paper_id(self, paper_id: str) -> bool:
        """验证paper_id格式"""
        patterns = [
            r'^PMC\d+$',                    # PMC格式
            r'^arXiv:\d{4}\.\d{4,5}(v\d+)?$',  # arXiv格式
            r'^[a-zA-Z0-9_-]+$'             # 通用格式
        ]
        return any(re.match(pattern, paper_id) for pattern in patterns)
    
    def _validate_doi(self, doi: str) -> bool:
        """验证DOI格式"""
        return bool(re.match(r'^10\.\d{4,9}/[-._;()\/:a-zA-Z0-9]+$', doi))
    
    def _validate_date(self, date: str) -> bool:
        """验证日期格式"""
        return bool(re.match(r'^\d{4}(-\d{2}(-\d{2})?)?$', date))
    
    def _validate_bbox_coords(self, bbox: BBox, width: int, height: int) -> bool:
        """验证边界框坐标"""
        return (
            0 <= bbox.x1 < bbox.x2 <= width and
            0 <= bbox.y1 < bbox.y2 <= height
        )
    
    def _validate_image_path(self, path: str) -> bool:
        """验证图片路径"""
        # 必须是相对路径
        if Path(path).is_absolute():
            return False
        # 必须是图片格式
        return path.lower().endswith(('.png', '.jpg', '.jpeg'))
    
    def _check_sections_consistency(self, sections):
        """检查章节一致性"""
        # 检查章节层级
        levels = [s.level for s in sections]
        if not levels:
            return
        
        # 层级应该从1开始
        if min(levels) != 1:
            self.warnings.append("章节层级应从1开始")
        
        # 检查层级跳跃
        for i in range(1, len(levels)):
            if levels[i] > levels[i-1] + 1:
                self.warnings.append(f"章节层级跳跃: {levels[i-1]} -> {levels[i]}")
    
    def _check_author_affiliations(self, authors, affiliations):
        """检查作者单位引用"""
        valid_ids = set(range(len(affiliations)))
        
        for author in authors:
            for aff_id in author.affiliation_ids:
                if aff_id not in valid_ids:
                    self.errors.append(
                        f"作者{author.name}引用了无效的单位ID: {aff_id}"
                    )
    
    def _check_figure_type_consistency(self, annotation: BBoxAnnotation):
        """检查图表类型一致性"""
        # 检查caption与类型的一致性
        if annotation.caption:
            caption_lower = annotation.caption.lower()
            
            if annotation.figure_type == FigureType.TABLE:
                if 'figure' in caption_lower and 'table' not in caption_lower:
                    self.warnings.append("标注为表格但caption包含'figure'")
            elif annotation.figure_type == FigureType.FIGURE:
                if 'table' in caption_lower and 'figure' not in caption_lower:
                    self.warnings.append("标注为图表但caption包含'table'")
    
    def _check_variable_axis_consistency(self, variables, axis):
        """检查变量与坐标轴一致性"""
        # 收集X/Y变量
        x_vars = [v for v in variables if v.role == "x"]
        y_vars = [v for v in variables if v.role == "y"]
        
        # 检查与坐标轴标签的一致性
        if axis.x_label and x_vars:
            # 简单的名称匹配
            if not any(v.name.lower() in axis.x_label.lower() for v in x_vars):
                self.warnings.append("X轴变量名与轴标签可能不一致")
        
        if axis.y_label and y_vars:
            if not any(v.name.lower() in axis.y_label.lower() for v in y_vars):
                self.warnings.append("Y轴变量名与轴标签可能不一致")
    
    def _check_key_findings(self, key_findings: str):
        """检查关键发现"""
        # 检查长度
        char_count = len(key_findings) + len(re.findall(r'[\u4e00-\u9fa5]', key_findings))
        if char_count > 100:
            self.warnings.append("key_findings过长")
        
        # 检查推断性词汇
        inference_words = [
            '可能', '也许', '或许', '大概', '推测',
            'might', 'maybe', 'perhaps', 'probably', 'possibly'
        ]
        
        for word in inference_words:
            if word in key_findings.lower():
                self.errors.append(f"key_findings包含推断性词汇: {word}")
    
    def _validate_conversations(self, conversations: List[Dict]) -> bool:
        """验证对话格式"""
        if not conversations:
            return False
        
        # 应该是human-gpt交替
        expected_from = "human"
        for conv in conversations:
            if conv.get("from") != expected_from:
                return False
            expected_from = "gpt" if expected_from == "human" else "human"
        
        # 最后一个应该是gpt
        return conversations[-1].get("from") == "gpt"
    
    def _check_grounding_coords(
        self, 
        text: str, 
        width: Union[int, List[int]], 
        height: Union[int, List[int]]
    ):
        """检查grounding坐标"""
        # 提取所有<box>标记
        box_pattern = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, text)
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 确定对应的图片尺寸
            if isinstance(width, list):
                # 多图情况，假设按顺序对应
                img_width = width[min(i, len(width)-1)]
                img_height = height[min(i, len(height)-1)]
            else:
                img_width = width
                img_height = height
            
            # 检查坐标范围
            if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
                self.errors.append(
                    f"Grounding坐标超出图片范围: [{x1},{y1},{x2},{y2}], "
                    f"图片尺寸: {img_width}x{img_height}"
                )
    
    def generate_report(self) -> str:
        """生成检查报告"""
        report = []
        
        if self.errors:
            report.append("=== 错误 ===")
            for error in self.errors:
                report.append(f"❌ {error}")
            report.append("")
        
        if self.warnings:
            report.append("=== 警告 ===")
            for warning in self.warnings:
                report.append(f"⚠️ {warning}")
            report.append("")
        
        if not self.errors and not self.warnings:
            report.append("✅ 所有检查通过")
        
        return "\n".join(report)