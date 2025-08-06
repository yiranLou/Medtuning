"""工作版本的增强图表检测器"""
import fitz
from pathlib import Path
from typing import List, Dict, Any
import logging

from ..schemas import BBox, FigureType
from dataclasses import dataclass
from typing import Optional

@dataclass
class DetectedFigure:
    """检测到的图表"""
    page_index: int
    bbox: BBox
    figure_type: FigureType
    caption: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

logger = logging.getLogger(__name__)


class WorkingEnhancedDetector:
    """可工作的增强检测器 - 专注于检测所有嵌入图片"""
    
    def __init__(self):
        self.min_figure_area = 5000  # 最小图片面积
        self.page_dpi = 200  # 与renderer保持一致的DPI
    
    def detect_all_elements(self, pdf_path: Path) -> Dict[str, List[DetectedFigure]]:
        """检测PDF中的所有图片和表格"""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        
        results = {
            'figures': [],
            'tables': [],
            'equations': []
        }
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 1. 检测嵌入的图片
                images = self._detect_embedded_images(page, page_num)
                results['figures'].extend(images)
                
                # 2. 检测基于文本的表格（简化版）
                tables = self._detect_text_tables(page, page_num)
                results['tables'].extend(tables)
            
            logger.info(f"检测完成: {len(results['figures'])}个图片, "
                       f"{len(results['tables'])}个表格")
        
        finally:
            doc.close()
        
        return results
    
    def _detect_embedded_images(self, page, page_num: int) -> List[DetectedFigure]:
        """检测页面中的嵌入图片"""
        figures = []
        
        # 获取所有嵌入的图片
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                
                # 获取图片在页面上的位置
                img_rects = page.get_image_rects(xref)
                
                for rect_idx, rect in enumerate(img_rects):
                    # 转换PDF坐标到page_dpi坐标
                    scale = self.page_dpi / 72.0
                    
                    # 创建边界框（转换为page_dpi坐标系）
                    bbox = BBox(
                        x1=max(0, int(rect.x0 * scale)),
                        y1=max(0, int(rect.y0 * scale)),
                        x2=max(int(rect.x0 * scale) + 1, int(rect.x1 * scale)),
                        y2=max(int(rect.y0 * scale) + 1, int(rect.y1 * scale))
                    )
                    
                    # 检查面积
                    area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
                    if area < self.min_figure_area:
                        continue
                    
                    # 查找图片标题
                    caption = self._find_figure_caption(page, bbox)
                    
                    figures.append(DetectedFigure(
                        page_index=page_num,
                        bbox=bbox,
                        figure_type=FigureType.FIGURE,
                        caption=caption or f"Figure {page_num + 1}-{img_index + 1}",
                        confidence=0.95,
                        metadata={
                            'xref': xref,
                            'rect_index': rect_idx
                        }
                    ))
                    
                    logger.debug(f"页面 {page_num + 1}: 检测到图片 {img_index + 1}, "
                               f"位置: ({bbox.x1}, {bbox.y1}) - ({bbox.x2}, {bbox.y2})")
            
            except Exception as e:
                logger.warning(f"处理图片 {img_index} 失败: {e}")
        
        return figures
    
    def _detect_text_tables(self, page, page_num: int) -> List[DetectedFigure]:
        """检测基于文本的表格"""
        tables = []
        
        # 获取页面文本块
        blocks = page.get_text("dict")
        
        # 查找包含"Table"关键词的区域
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # 文本块
                block_text = self._extract_block_text(block)
                
                # 简单检查是否可能是表格
                if self._is_likely_table(block_text):
                    bbox_coords = block.get("bbox", [0, 0, 1, 1])
                    
                    # 转换PDF坐标到page_dpi坐标
                    scale = self.page_dpi / 72.0
                    
                    bbox = BBox(
                        x1=max(0, int(bbox_coords[0] * scale)),
                        y1=max(0, int(bbox_coords[1] * scale)),
                        x2=max(int(bbox_coords[0] * scale) + 1, int(bbox_coords[2] * scale)),
                        y2=max(int(bbox_coords[1] * scale) + 1, int(bbox_coords[3] * scale))
                    )
                    
                    # 提取表格标题
                    caption = self._extract_table_caption(block_text)
                    
                    if caption:  # 只有有标题的才认为是表格
                        tables.append(DetectedFigure(
                            page_index=page_num,
                            bbox=bbox,
                            figure_type=FigureType.TABLE,
                            caption=caption,
                            confidence=0.7,
                            metadata={'text_based': True}
                        ))
        
        return tables
    
    def _find_figure_caption(self, page, figure_bbox: BBox) -> str:
        """查找图片的标题"""
        # 将page_dpi坐标转换回PDF坐标进行文本搜索
        scale = self.page_dpi / 72.0
        
        # 在图片下方查找文本
        search_area = fitz.Rect(
            (figure_bbox.x1 / scale) - 20,
            figure_bbox.y2 / scale,
            (figure_bbox.x2 / scale) + 20,
            min((figure_bbox.y2 / scale) + 100, page.rect.height)
        )
        
        text = page.get_textbox(search_area)
        
        # 查找Figure关键词
        lines = text.strip().split('\n')
        for line in lines[:3]:  # 只看前3行
            if any(keyword in line for keyword in ['Figure', 'Fig.', 'FIGURE']):
                return line.strip()
        
        return None
    
    def _extract_block_text(self, block: Dict) -> str:
        """提取文本块的文本"""
        text_parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))
        return " ".join(text_parts)
    
    def _is_likely_table(self, text: str) -> bool:
        """判断文本是否可能是表格"""
        # 检查是否包含表格关键词
        table_keywords = ['Table', 'TABLE', 'Tab.']
        has_keyword = any(keyword in text for keyword in table_keywords)
        
        # 检查是否有表格特征（多个数字、分隔符等）
        has_numbers = text.count(' ') > 5 and any(c.isdigit() for c in text)
        
        return has_keyword or has_numbers
    
    def _extract_table_caption(self, text: str) -> str:
        """提取表格标题"""
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['Table', 'TABLE', 'Tab.']):
                return line.strip()
        return None