"""图表检测与提取模块"""
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from PIL import Image

from ..schemas import BBox, FigureType

logger = logging.getLogger(__name__)


@dataclass
class DetectedFigure:
    """检测到的图表"""
    page_index: int
    bbox: BBox
    figure_type: FigureType
    caption: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def iou(self, other: 'DetectedFigure') -> float:
        """计算与另一个图表的IoU"""
        if self.page_index != other.page_index:
            return 0.0
        
        # 计算交集
        x1 = max(self.bbox.x1, other.bbox.x1)
        y1 = max(self.bbox.y1, other.bbox.y1)
        x2 = min(self.bbox.x2, other.bbox.x2)
        y2 = min(self.bbox.y2, other.bbox.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (self.bbox.x2 - self.bbox.x1) * (self.bbox.y2 - self.bbox.y1)
        area2 = (other.bbox.x2 - other.bbox.x1) * (other.bbox.y2 - other.bbox.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class FigureDetector:
    """图表检测器基类"""
    
    def detect(self, pdf_path: Path) -> List[DetectedFigure]:
        """检测PDF中的图表"""
        raise NotImplementedError
    
    def _deduplicate_figures(
        self, 
        figures: List[DetectedFigure], 
        iou_threshold: float = 0.9
    ) -> List[DetectedFigure]:
        """去重：IoU > threshold的保留面积更大者"""
        if not figures:
            return []
        
        # 按面积降序排序
        figures = sorted(figures, key=lambda f: 
                        (f.bbox.x2 - f.bbox.x1) * (f.bbox.y2 - f.bbox.y1), 
                        reverse=True)
        
        kept = []
        for fig in figures:
            # 检查是否与已保留的图表重叠
            is_duplicate = False
            for kept_fig in kept:
                if fig.iou(kept_fig) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(fig)
        
        return kept


class PDFFigures2Detector(FigureDetector):
    """使用PDFFigures2进行图表检测"""
    
    def __init__(self, pdffigures2_path: Optional[str] = None):
        self.pdffigures2_path = pdffigures2_path or "pdffigures2"
        self._check_availability()
    
    def _check_availability(self):
        """检查PDFFigures2是否可用"""
        try:
            result = subprocess.run(
                [self.pdffigures2_path, "--help"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("PDFFigures2不可用")
        except FileNotFoundError:
            raise RuntimeError(
                "未找到PDFFigures2。请安装: "
                "https://github.com/allenai/pdffigures2"
            )
    
    def detect(self, pdf_path: Path, dpi: int = 200) -> List[DetectedFigure]:
        """使用PDFFigures2检测图表"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 准备输出目录
        output_dir = pdf_path.parent / f"{pdf_path.stem}_pdffigures2"
        output_dir.mkdir(exist_ok=True)
        
        # 运行PDFFigures2
        cmd = [
            self.pdffigures2_path,
            "-g", str(output_dir),  # 保存图片
            "-d", str(dpi),  # DPI设置
            "-j", str(output_dir / "data.json"),  # JSON输出
            str(pdf_path)
        ]
        
        logger.info(f"运行PDFFigures2: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"PDFFigures2错误: {result.stderr}")
            return []
        
        # 解析结果
        json_file = output_dir / "data.json"
        if not json_file.exists():
            logger.warning("PDFFigures2未生成输出文件")
            return []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        figures = []
        for item in data:
            # 提取图表信息
            for fig in item.get('figures', []):
                bbox = self._convert_bbox(fig['regionBoundary'], dpi)
                figure_type = self._infer_figure_type(fig)
                
                detected = DetectedFigure(
                    page_index=fig['page'],
                    bbox=bbox,
                    figure_type=figure_type,
                    caption=fig.get('caption', ''),
                    confidence=1.0,
                    metadata={
                        'figure_type': fig.get('figType', 'Figure'),
                        'name': fig.get('name', '')
                    }
                )
                figures.append(detected)
        
        # 去重
        figures = self._deduplicate_figures(figures)
        logger.info(f"PDFFigures2检测到{len(figures)}个图表")
        
        return figures
    
    def _convert_bbox(self, region: Dict, dpi: int) -> BBox:
        """转换PDFFigures2的坐标到像素坐标"""
        # PDFFigures2使用PDF点坐标
        scale = dpi / 72.0
        return BBox(
            x1=int(region['x1'] * scale),
            y1=int(region['y1'] * scale),
            x2=int(region['x2'] * scale),
            y2=int(region['y2'] * scale)
        )
    
    def _infer_figure_type(self, fig: Dict) -> FigureType:
        """推断图表类型"""
        fig_type = fig.get('figType', '').lower()
        name = fig.get('name', '').lower()
        
        if 'table' in fig_type or 'table' in name:
            return FigureType.TABLE
        elif 'equation' in fig_type or 'eq' in name:
            return FigureType.EQUATION
        elif 'diagram' in name:
            return FigureType.DIAGRAM
        elif 'flow' in name:
            return FigureType.FLOWCHART
        else:
            return FigureType.FIGURE


class HeuristicDetector(FigureDetector):
    """基于启发式规则的图表检测（备用方案）"""
    
    def __init__(self):
        self.figure_keywords = [
            'Figure', 'Fig.', 'Table', 'Tab.', 'Scheme', 
            'Chart', 'Graph', 'Diagram', 'Plot'
        ]
    
    def detect(self, pdf_path: Path, images: List[Image.Image] = None) -> List[DetectedFigure]:
        """使用启发式方法检测图表"""
        # 这是一个简化的实现，实际应用中需要更复杂的算法
        logger.warning("使用启发式检测器，结果可能不准确")
        
        figures = []
        
        # TODO: 实现基于以下策略的检测
        # 1. 检测大块的非文本区域
        # 2. 寻找图表标题模式（Figure 1:, Table 1:等）
        # 3. 使用边缘检测找到矩形框
        # 4. 检测表格线条模式
        
        return figures


class CombinedDetector(FigureDetector):
    """组合多个检测器"""
    
    def __init__(self, detectors: List[FigureDetector]):
        self.detectors = detectors
    
    def detect(self, pdf_path: Path) -> List[DetectedFigure]:
        """使用所有检测器并合并结果"""
        all_figures = []
        
        for detector in self.detectors:
            try:
                figures = detector.detect(pdf_path)
                all_figures.extend(figures)
                logger.info(f"{detector.__class__.__name__}检测到{len(figures)}个图表")
            except Exception as e:
                logger.error(f"{detector.__class__.__name__}检测失败: {e}")
        
        # 合并和去重
        merged = self._merge_detections(all_figures)
        return merged
    
    def _merge_detections(
        self, 
        figures: List[DetectedFigure], 
        iou_threshold: float = 0.8
    ) -> List[DetectedFigure]:
        """合并多个检测器的结果"""
        if not figures:
            return []
        
        # 按置信度降序排序
        figures = sorted(figures, key=lambda f: f.confidence, reverse=True)
        
        merged = []
        for fig in figures:
            # 检查是否与已合并的图表重叠
            should_merge = False
            for i, merged_fig in enumerate(merged):
                if fig.iou(merged_fig) > iou_threshold:
                    # 合并信息
                    if fig.caption and not merged_fig.caption:
                        merged[i].caption = fig.caption
                    if fig.confidence > merged_fig.confidence:
                        merged[i].confidence = fig.confidence
                    should_merge = True
                    break
            
            if not should_merge:
                merged.append(fig)
        
        return merged


def create_detector(use_pdffigures2: bool = True) -> FigureDetector:
    """创建图表检测器"""
    detectors = []
    
    if use_pdffigures2:
        try:
            pdffigures2 = PDFFigures2Detector()
            detectors.append(pdffigures2)
        except RuntimeError as e:
            logger.warning(f"PDFFigures2不可用: {e}")
    
    # 添加备用检测器
    if not detectors:
        detectors.append(HeuristicDetector())
    
    if len(detectors) == 1:
        return detectors[0]
    else:
        return CombinedDetector(detectors)