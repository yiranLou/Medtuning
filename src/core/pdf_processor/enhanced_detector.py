"""增强的图表和表格检测器"""
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
import fitz  # PyMuPDF

from ..schemas import BBox, FigureType
from .detector import DetectedFigure

logger = logging.getLogger(__name__)


class EnhancedFigureTableDetector:
    """增强的图表和表格检测器"""
    
    def __init__(self):
        self.min_figure_area = 10000  # 最小图表面积
        self.min_table_area = 5000   # 最小表格面积
    
    def detect_all_elements(self, pdf_path: Path) -> Dict[str, List[DetectedFigure]]:
        """检测PDF中的所有图表和表格"""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        
        results = {
            'figures': [],
            'tables': [],
            'equations': []
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 1. 检测图片
            images = self._detect_images(page, page_num)
            results['figures'].extend(images)
            
            # 2. 检测表格
            tables = self._detect_tables(page, page_num)
            results['tables'].extend(tables)
            
            # 3. 检测公式
            equations = self._detect_equations(page, page_num)
            results['equations'].extend(equations)
        
        doc.close()
        
        logger.info(f"检测结果: {len(results['figures'])}个图表, "
                   f"{len(results['tables'])}个表格, {len(results['equations'])}个公式")
        
        return results
    
    def _detect_images(self, page, page_num: int) -> List[DetectedFigure]:
        """检测页面中的图片"""
        images = []
        
        # 方法1: 从PDF中提取嵌入的图片
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                # 获取图片的边界框
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # 查找图片在页面上的位置
                img_rects = page.get_image_rects(xref)
                
                for rect in img_rects:
                    bbox = BBox(
                        x1=int(rect.x0),
                        y1=int(rect.y0),
                        x2=int(rect.x1),
                        y2=int(rect.y1)
                    )
                    
                    # 检查面积
                    area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
                    if area < self.min_figure_area:
                        continue
                    
                    # 查找相关的caption
                    caption = self._find_caption(page, bbox)
                    
                    images.append(DetectedFigure(
                        page_index=page_num,
                        bbox=bbox,
                        figure_type=FigureType.FIGURE,
                        caption=caption,
                        confidence=0.9
                    ))
                
                pix = None  # 释放资源
                
            except Exception as e:
                logger.warning(f"处理图片{img_index}失败: {e}")
        
        # 方法2: 通过边缘检测找到可能的图表区域
        additional_figures = self._detect_figure_regions(page, page_num)
        images.extend(additional_figures)
        
        return images
    
    def _detect_tables(self, page, page_num: int) -> List[DetectedFigure]:
        """检测表格"""
        tables = []
        
        # 获取页面文本和布局
        blocks = page.get_text("dict")
        
        # 方法1: 查找表格线条
        table_regions = self._find_table_lines(page)
        
        for region in table_regions:
            # 验证是否真的是表格
            if self._is_table_region(blocks, region):
                caption = self._find_caption(page, region)
                
                tables.append(DetectedFigure(
                    page_index=page_num,
                    bbox=region,
                    figure_type=FigureType.TABLE,
                    caption=caption,
                    confidence=0.85,
                    metadata={'has_grid': True}
                ))
        
        # 方法2: 通过文本模式检测表格（无边框表格）
        text_tables = self._detect_text_tables(blocks, page_num)
        tables.extend(text_tables)
        
        return tables
    
    def _detect_equations(self, page, page_num: int) -> List[DetectedFigure]:
        """检测数学公式"""
        equations = []
        
        # 查找包含数学符号的区域
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # 文本块
                for line in block.get("lines", []):
                    text = self._extract_line_text(line)
                    
                    # 简单的公式检测规则
                    if self._is_equation(text):
                        bbox = self._get_line_bbox(line)
                        
                        equations.append(DetectedFigure(
                            page_index=page_num,
                            bbox=bbox,
                            figure_type=FigureType.EQUATION,
                            caption=text[:100],  # 前100字符作为预览
                            confidence=0.7
                        ))
        
        return equations
    
    def _find_table_lines(self, page) -> List[BBox]:
        """通过线条检测找到表格区域"""
        # 使用PyMuPDF的绘图命令查找线条
        drawings = page.get_drawings()
        
        # 收集所有线条
        lines = []
        for item in drawings:
            if 'items' in item:
                for subitem in item['items']:
                    if subitem.get('type') == 'l':  # line
                        p1 = subitem.get('p1')
                        p2 = subitem.get('p2')
                        if p1 and p2:
                            lines.append([
                                [int(p1.x), int(p1.y), int(p2.x), int(p2.y)]
                            ])
        
        if not lines:
            return []
        
        # 转换为numpy数组格式
        lines_array = np.array(lines)
        
        # 查找矩形区域
        table_regions = self._find_rectangular_regions(lines_array, 
                                                      int(page.rect.width), 
                                                      int(page.rect.height))
        
        return table_regions
    
    def _detect_figure_regions(self, page, page_num: int) -> List[DetectedFigure]:
        """通过区域检测找到可能的图表"""
        figures = []
        
        # 获取页面的所有绘图命令
        drawings = page.get_drawings()
        
        # 分组相近的绘图元素
        grouped_drawings = self._group_drawings(drawings)
        
        for group in grouped_drawings:
            bbox = self._get_group_bbox(group)
            area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            
            if area >= self.min_figure_area:
                # 检查是否包含图表特征
                if self._has_figure_characteristics(group):
                    caption = self._find_caption(page, bbox)
                    
                    figures.append(DetectedFigure(
                        page_index=page_num,
                        bbox=bbox,
                        figure_type=FigureType.FIGURE,
                        caption=caption,
                        confidence=0.75
                    ))
        
        return figures
    
    def _detect_text_tables(self, blocks: Dict, page_num: int) -> List[DetectedFigure]:
        """检测基于文本对齐的表格"""
        tables = []
        
        # 分析文本块的对齐模式
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # 文本块
                lines = block.get("lines", [])
                
                # 检查是否有表格式的对齐
                if self._has_table_alignment(lines):
                    bbox = self._get_block_bbox(block)
                    
                    tables.append(DetectedFigure(
                        page_index=page_num,
                        bbox=bbox,
                        figure_type=FigureType.TABLE,
                        caption=self._extract_table_caption(block),
                        confidence=0.6,
                        metadata={'has_grid': False}
                    ))
        
        return tables
    
    def _find_caption(self, page, bbox: BBox) -> Optional[str]:
        """查找图表的标题"""
        # 在图表上方和下方查找文本
        search_regions = [
            # 上方
            BBox(bbox.x1 - 50, bbox.y1 - 100, bbox.x2 + 50, bbox.y1),
            # 下方
            BBox(bbox.x1 - 50, bbox.y2, bbox.x2 + 50, bbox.y2 + 100)
        ]
        
        for region in search_regions:
            text = page.get_textbox(fitz.Rect(
                region.x1, region.y1, region.x2, region.y2
            ))
            
            # 查找Figure/Table关键词
            if any(keyword in text for keyword in 
                   ['Figure', 'Fig.', 'Table', 'Tab.', '图', '表']):
                return text.strip()
        
        return None
    
    def _is_equation(self, text: str) -> bool:
        """判断文本是否可能是数学公式"""
        # 数学符号
        math_symbols = ['∫', '∑', '∏', '√', '∞', '∈', '∀', '∃', '⊂', '⊃', 
                       '∪', '∩', '≤', '≥', '≠', '≈', '∝', '∂', '∇']
        
        # LaTeX命令
        latex_patterns = [r'\\[a-zA-Z]+', r'\^', r'_', r'\\frac', r'\\sqrt']
        
        # 检查数学符号
        if any(symbol in text for symbol in math_symbols):
            return True
        
        # 检查LaTeX模式
        for pattern in latex_patterns:
            if re.search(pattern, text):
                return True
        
        # 检查公式模式 (如 x = y + z)
        if re.search(r'[a-zA-Z]\s*=\s*[a-zA-Z\d\+\-\*/\(\)]+', text):
            return True
        
        return False
    
    def _is_table_region(self, blocks: Dict, region: BBox) -> bool:
        """验证区域是否包含表格内容"""
        # 检查区域内的文本模式
        text_in_region = []
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # 文本块
                block_bbox = self._get_block_bbox(block)
                
                # 检查是否在区域内
                if self._bbox_overlap(block_bbox, region):
                    for line in block.get("lines", []):
                        text = self._extract_line_text(line)
                        text_in_region.append(text)
        
        # 分析文本模式
        if len(text_in_region) < 2:
            return False
        
        # 检查是否有表格特征
        # 1. 多列对齐
        # 2. 重复的分隔符
        # 3. 数字模式
        
        return self._has_table_pattern(text_in_region)
    
    def _has_table_pattern(self, lines: List[str]) -> bool:
        """检查文本行是否有表格模式"""
        if len(lines) < 2:
            return False
        
        # 检查分隔符
        separator_count = 0
        for line in lines:
            if any(sep in line for sep in ['|', '\t', '  ']):
                separator_count += 1
        
        # 如果大部分行都有分隔符，可能是表格
        if separator_count / len(lines) > 0.5:
            return True
        
        # 检查对齐模式
        # TODO: 实现更复杂的对齐检测
        
        return False
    
    def _find_rectangular_regions(self, lines: np.ndarray, width: int, height: int) -> List[BBox]:
        """通过线条查找矩形区域"""
        # 分离水平和垂直线
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # 水平线
                horizontal_lines.append((min(x1, x2), y1, max(x1, x2), y1))
            elif abs(x2 - x1) < 5:  # 垂直线
                vertical_lines.append((x1, min(y1, y2), x1, max(y1, y2)))
        
        # 合并相近的线条
        h_lines = self._merge_lines(horizontal_lines, 'horizontal')
        v_lines = self._merge_lines(vertical_lines, 'vertical')
        
        # 查找矩形
        rectangles = []
        for i, h1 in enumerate(h_lines):
            for j, h2 in enumerate(h_lines[i+1:], i+1):
                for v1 in v_lines:
                    for v2 in v_lines:
                        if v1[0] < v2[0]:  # v1在v2左边
                            # 检查是否形成矩形
                            if (self._lines_intersect(h1, v1) and 
                                self._lines_intersect(h1, v2) and
                                self._lines_intersect(h2, v1) and 
                                self._lines_intersect(h2, v2)):
                                
                                bbox = BBox(
                                    x1=int(v1[0]),
                                    y1=int(h1[1]),
                                    x2=int(v2[0]),
                                    y2=int(h2[1])
                                )
                                
                                # 检查面积
                                area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
                                if area >= self.min_table_area:
                                    rectangles.append(bbox)
        
        # 移除重复和嵌套的矩形
        rectangles = self._remove_nested_rectangles(rectangles)
        
        return rectangles
    
    def _merge_lines(self, lines: List[Tuple], direction: str) -> List[Tuple]:
        """合并相近的线条"""
        if not lines:
            return []
        
        merged = []
        lines = sorted(lines)
        
        current = list(lines[0])
        for line in lines[1:]:
            if direction == 'horizontal':
                # 检查y坐标是否相近
                if abs(line[1] - current[1]) < 5:
                    # 扩展x范围
                    current[0] = min(current[0], line[0])
                    current[2] = max(current[2], line[2])
                else:
                    merged.append(tuple(current))
                    current = list(line)
            else:  # vertical
                # 检查x坐标是否相近
                if abs(line[0] - current[0]) < 5:
                    # 扩展y范围
                    current[1] = min(current[1], line[1])
                    current[3] = max(current[3], line[3])
                else:
                    merged.append(tuple(current))
                    current = list(line)
        
        merged.append(tuple(current))
        return merged
    
    def _lines_intersect(self, h_line: Tuple, v_line: Tuple) -> bool:
        """检查水平线和垂直线是否相交"""
        # h_line: (x1, y, x2, y)
        # v_line: (x, y1, x, y2)
        return (h_line[0] <= v_line[0] <= h_line[2] and 
                v_line[1] <= h_line[1] <= v_line[3])
    
    def _remove_nested_rectangles(self, rectangles: List[BBox]) -> List[BBox]:
        """移除嵌套的矩形"""
        if not rectangles:
            return []
        
        # 按面积降序排序
        rectangles = sorted(rectangles, 
                          key=lambda r: (r.x2 - r.x1) * (r.y2 - r.y1), 
                          reverse=True)
        
        result = []
        for rect in rectangles:
            # 检查是否被已有矩形包含
            is_nested = False
            for existing in result:
                if (rect.x1 >= existing.x1 and rect.y1 >= existing.y1 and
                    rect.x2 <= existing.x2 and rect.y2 <= existing.y2):
                    is_nested = True
                    break
            
            if not is_nested:
                result.append(rect)
        
        return result
    
    def _group_drawings(self, drawings: List[Dict]) -> List[List[Dict]]:
        """分组相近的绘图元素"""
        if not drawings:
            return []
        
        groups = []
        used = set()
        
        for i, drawing in enumerate(drawings):
            if i in used:
                continue
            
            group = [drawing]
            used.add(i)
            
            # 查找相近的元素
            for j, other in enumerate(drawings[i+1:], i+1):
                if j not in used and self._drawings_nearby(drawing, other):
                    group.append(other)
                    used.add(j)
            
            if group:
                groups.append(group)
        
        return groups
    
    def _drawings_nearby(self, d1: Dict, d2: Dict, threshold: float = 20) -> bool:
        """检查两个绘图元素是否相近"""
        # 获取边界
        bbox1 = self._get_drawing_bbox(d1)
        bbox2 = self._get_drawing_bbox(d2)
        
        # 检查距离
        x_dist = min(abs(bbox1.x2 - bbox2.x1), abs(bbox2.x2 - bbox1.x1))
        y_dist = min(abs(bbox1.y2 - bbox2.y1), abs(bbox2.y2 - bbox1.y1))
        
        return x_dist < threshold and y_dist < threshold
    
    def _get_drawing_bbox(self, drawing: Dict) -> BBox:
        """获取单个绘图元素的边界框"""
        if 'rect' in drawing:
            rect = drawing['rect']
            return BBox(
                x1=int(rect.x0),
                y1=int(rect.y0),
                x2=int(rect.x1),
                y2=int(rect.y1)
            )
        
        # 处理路径
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for item in drawing.get('items', []):
            if 'p' in item:
                point = item['p']
                min_x = min(min_x, point.x)
                min_y = min(min_y, point.y)
                max_x = max(max_x, point.x)
                max_y = max(max_y, point.y)
        
        return BBox(
            x1=int(min_x) if min_x != float('inf') else 0,
            y1=int(min_y) if min_y != float('inf') else 0,
            x2=int(max_x) if max_x != float('-inf') else 0,
            y2=int(max_y) if max_y != float('-inf') else 0
        )
    
    def _get_group_bbox(self, group: List[Dict]) -> BBox:
        """获取组的边界框"""
        if not group:
            return BBox(x1=0, y1=0, x2=0, y2=0)
        
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for item in group:
            bbox = self._get_drawing_bbox(item)
            min_x = min(min_x, bbox.x1)
            min_y = min(min_y, bbox.y1)
            max_x = max(max_x, bbox.x2)
            max_y = max(max_y, bbox.y2)
        
        return BBox(
            x1=int(min_x),
            y1=int(min_y),
            x2=int(max_x),
            y2=int(max_y)
        )
    
    def _has_figure_characteristics(self, group: List[Dict]) -> bool:
        """检查是否具有图表特征"""
        if not group:
            return False
        
        # 统计不同类型的元素
        has_curves = False
        has_fills = False
        has_text = False
        element_count = 0
        
        for item in group:
            element_count += 1
            
            if 'type' in item:
                if item['type'] == 'f':  # fill
                    has_fills = True
                elif item['type'] == 's':  # stroke
                    # 检查是否有曲线
                    if 'items' in item:
                        for subitem in item['items']:
                            if subitem.get('type') == 'c':  # curve
                                has_curves = True
                elif item['type'] == 't':  # text
                    has_text = True
        
        # 图表通常有多个元素，可能包含曲线或填充
        return element_count > 5 and (has_curves or has_fills)
    
    def _has_table_alignment(self, lines: List[Dict]) -> bool:
        """检查是否有表格式的对齐"""
        if len(lines) < 3:
            return False
        
        # 收集每行的x坐标
        x_positions = []
        for line in lines:
            spans_x = []
            for span in line.get("spans", []):
                bbox = span.get("bbox", [0, 0, 0, 0])
                spans_x.append(bbox[0])
            
            if spans_x:
                x_positions.append(sorted(spans_x))
        
        # 检查是否有共同的x坐标（列对齐）
        if not x_positions:
            return False
        
        # 统计共同的x位置
        common_x_count = 0
        for i in range(len(x_positions) - 1):
            for x in x_positions[i]:
                # 检查下一行是否有相近的x位置
                for next_x in x_positions[i + 1]:
                    if abs(x - next_x) < 5:  # 5像素容差
                        common_x_count += 1
                        break
        
        # 如果有足够多的对齐，认为是表格
        return common_x_count > len(lines) * 0.5
    
    def _extract_table_caption(self, block: Dict) -> Optional[str]:
        """提取表格标题"""
        # 查找包含Table关键字的行
        for line in block.get("lines", []):
            text = self._extract_line_text(line)
            if any(keyword in text for keyword in ['Table', 'Tab.', '表']):
                return text.strip()
        
        # 如果没找到，返回第一行作为标题
        if block.get("lines"):
            return self._extract_line_text(block["lines"][0]).strip()
        
        return None
    
    def _extract_line_text(self, line: Dict) -> str:
        """提取行文本"""
        text_parts = []
        for span in line.get("spans", []):
            text_parts.append(span.get("text", ""))
        return " ".join(text_parts)
    
    def _get_line_bbox(self, line: Dict) -> BBox:
        """获取行的边界框"""
        bbox = line.get("bbox", [0, 0, 0, 0])
        return BBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3])
        )
    
    def _get_block_bbox(self, block: Dict) -> BBox:
        """获取块的边界框"""
        bbox = block.get("bbox", [0, 0, 0, 0])
        return BBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3])
        )
    
    def _bbox_overlap(self, bbox1: BBox, bbox2: BBox) -> bool:
        """检查两个边界框是否重叠"""
        return not (bbox1.x2 < bbox2.x1 or bbox1.x1 > bbox2.x2 or
                   bbox1.y2 < bbox2.y1 or bbox1.y1 > bbox2.y2)