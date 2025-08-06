"""增强版边界框注释器 - 利用完整图表提取"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from PIL import Image
import numpy as np

from ..schemas import DetectedFigure, FigureType
from ..pdf_processor.renderer import PDFRenderer
from .base_annotator import BaseAnnotator
from .mistral_annotator import MistralBBoxAnnotator

logger = logging.getLogger(__name__)


class EnhancedBBoxAnnotator(BaseAnnotator):
    """增强版注释器 - 结合完整图像分析和OCR"""
    
    def __init__(self, api_key: str, use_ocr: bool = True):
        super().__init__()
        self.mistral_annotator = MistralBBoxAnnotator(api_key)
        self.use_ocr = use_ocr
        self.ocr_engine = None
        
        if use_ocr:
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                logger.info("OCR engine initialized")
            except ImportError:
                logger.warning("pytesseract not available, OCR features disabled")
                self.use_ocr = False
    
    async def annotate_element(self, element: DetectedFigure, 
                              pdf_path: Path,
                              context: Optional[str] = None) -> Dict[str, Any]:
        """增强的元素注释"""
        
        # 1. 使用Mistral进行基础注释
        base_annotation = await self.mistral_annotator.annotate_element(element, pdf_path, context)
        
        # 2. 提取并分析完整图像
        image_analysis = await self._analyze_full_image(element, pdf_path)
        
        # 3. 如果是表格，进行OCR
        if element.figure_type == FigureType.TABLE and self.use_ocr:
            ocr_results = await self._perform_ocr(element, pdf_path)
            if ocr_results:
                base_annotation['table_csv'] = ocr_results.get('csv_data', '')
                base_annotation['ocr_confidence'] = ocr_results.get('confidence', 0)
        
        # 4. 增强注释
        enhanced_annotation = self._enhance_annotation(base_annotation, image_analysis)
        
        return enhanced_annotation
    
    async def _analyze_full_image(self, element: DetectedFigure, pdf_path: Path) -> Dict[str, Any]:
        """分析完整图像内容"""
        analysis = {
            'has_multiple_panels': False,
            'color_scheme': [],
            'visual_elements': [],
            'complexity': 'medium'
        }
        
        try:
            # 渲染图像
            renderer = PDFRenderer(pdf_path)
            temp_path = Path(f"/tmp/temp_{element.page_index}_{element.bbox.x1}.png")
            renderer.crop_region(element.page_index, element.bbox, temp_path, use_high_dpi=True)
            
            # 加载图像进行分析
            img = Image.open(temp_path)
            img_array = np.array(img)
            
            # 分析图像特征
            analysis['image_size'] = img.size
            analysis['aspect_ratio'] = img.size[0] / img.size[1]
            
            # 检测是否有多个子图
            if self._detect_subplots(img_array):
                analysis['has_multiple_panels'] = True
            
            # 分析颜色
            analysis['color_scheme'] = self._analyze_colors(img_array)
            
            # 检测视觉元素
            analysis['visual_elements'] = self._detect_visual_elements(img_array)
            
            # 评估复杂度
            analysis['complexity'] = self._assess_complexity(img_array)
            
            # 清理临时文件
            temp_path.unlink(missing_ok=True)
            renderer.close()
            
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
        
        return analysis
    
    async def _perform_ocr(self, element: DetectedFigure, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """对表格进行OCR"""
        if not self.ocr_engine:
            return None
        
        try:
            # 渲染高分辨率图像
            renderer = PDFRenderer(pdf_path)
            temp_path = Path(f"/tmp/ocr_{element.page_index}_{element.bbox.x1}.png")
            renderer.crop_region(element.page_index, element.bbox, temp_path, use_high_dpi=True)
            
            # 执行OCR
            ocr_text = self.ocr_engine.image_to_string(str(temp_path))
            
            # 解析表格结构
            csv_data = self._parse_table_from_ocr(ocr_text)
            
            # 获取置信度
            ocr_data = self.ocr_engine.image_to_data(str(temp_path), output_type=self.ocr_engine.Output.DICT)
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # 清理
            temp_path.unlink(missing_ok=True)
            renderer.close()
            
            return {
                'raw_text': ocr_text,
                'csv_data': csv_data,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"OCR失败: {e}")
            return None
    
    def _enhance_annotation(self, base_annotation: Dict[str, Any], 
                          image_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """增强基础注释"""
        enhanced = base_annotation.copy()
        
        # 添加图像分析结果
        enhanced['image_analysis'] = image_analysis
        
        # 根据复杂度调整描述
        if image_analysis['complexity'] == 'high':
            enhanced['requires_detailed_analysis'] = True
        
        # 如果有多个面板，添加相关信息
        if image_analysis['has_multiple_panels']:
            enhanced['has_multiple_panels'] = True
            enhanced['panel_relationship'] = "Multiple related visualizations"
        
        # 增强变量描述
        if 'variables' in enhanced:
            enhanced['variables'] = self._enhance_variables(
                enhanced['variables'], 
                image_analysis['visual_elements']
            )
        
        return enhanced
    
    def _detect_subplots(self, img_array: np.ndarray) -> bool:
        """检测是否包含多个子图"""
        # 简单的启发式方法：检测内部边界
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        
        # 检测水平和垂直线
        h_lines = self._detect_lines(gray, axis=0)
        v_lines = self._detect_lines(gray, axis=1)
        
        # 如果有多条内部分割线，可能是多子图
        return len(h_lines) > 2 or len(v_lines) > 2
    
    def _detect_lines(self, gray_img: np.ndarray, axis: int) -> List[int]:
        """检测图像中的线条"""
        # 计算梯度
        if axis == 0:  # 水平线
            grad = np.abs(np.diff(gray_img, axis=0))
            line_scores = np.mean(grad, axis=1)
        else:  # 垂直线
            grad = np.abs(np.diff(gray_img, axis=1))
            line_scores = np.mean(grad, axis=0)
        
        # 找到高梯度位置
        threshold = np.mean(line_scores) + 2 * np.std(line_scores)
        lines = np.where(line_scores > threshold)[0]
        
        # 合并相近的线
        merged_lines = []
        for line in lines:
            if not merged_lines or line - merged_lines[-1] > 10:
                merged_lines.append(line)
        
        return merged_lines
    
    def _analyze_colors(self, img_array: np.ndarray) -> List[str]:
        """分析图像的颜色方案"""
        # 获取主要颜色
        pixels = img_array.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        color_categories = []
        if len(unique_colors) > 100:
            color_categories.append("multicolor")
        elif len(unique_colors) > 10:
            color_categories.append("color")
        else:
            color_categories.append("grayscale")
        
        # 检测特定颜色模式
        avg_color = np.mean(pixels, axis=0)
        if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            color_categories.append("red_dominant")
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            color_categories.append("green_dominant")
        elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            color_categories.append("blue_dominant")
        
        return color_categories
    
    def _detect_visual_elements(self, img_array: np.ndarray) -> List[str]:
        """检测视觉元素"""
        elements = []
        
        # 这里可以使用更复杂的计算机视觉技术
        # 简单示例：
        
        # 检测是否有网格
        if self._has_grid_pattern(img_array):
            elements.append("grid")
        
        # 检测是否有圆形元素（可能是散点图）
        if self._has_circular_elements(img_array):
            elements.append("circles")
        
        # 检测条形
        if self._has_bars(img_array):
            elements.append("bars")
        
        return elements
    
    def _has_grid_pattern(self, img_array: np.ndarray) -> bool:
        """检测网格模式"""
        # 简化实现
        gray = np.mean(img_array, axis=2)
        h_lines = self._detect_lines(gray, axis=0)
        v_lines = self._detect_lines(gray, axis=1)
        return len(h_lines) > 5 and len(v_lines) > 5
    
    def _has_circular_elements(self, img_array: np.ndarray) -> bool:
        """检测圆形元素"""
        # 简化实现 - 实际应使用霍夫圆检测
        return False
    
    def _has_bars(self, img_array: np.ndarray) -> bool:
        """检测条形图元素"""
        # 简化实现
        gray = np.mean(img_array, axis=2)
        # 检测垂直条形的特征
        col_std = np.std(gray, axis=0)
        return np.max(col_std) > np.mean(col_std) * 2
    
    def _assess_complexity(self, img_array: np.ndarray) -> str:
        """评估图像复杂度"""
        # 基于信息熵的简单评估
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        if entropy > 7:
            return "high"
        elif entropy > 5:
            return "medium"
        else:
            return "low"
    
    def _parse_table_from_ocr(self, ocr_text: str) -> str:
        """从OCR文本解析表格结构"""
        lines = ocr_text.strip().split('\n')
        
        # 简单的CSV转换
        csv_lines = []
        for line in lines:
            # 使用多个空格作为分隔符
            cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
            if cells:
                csv_lines.append(','.join(cells))
        
        return '\n'.join(csv_lines)
    
    def _enhance_variables(self, variables: List[Dict[str, Any]], 
                         visual_elements: List[str]) -> List[Dict[str, Any]]:
        """增强变量描述"""
        enhanced_vars = []
        
        for var in variables:
            enhanced_var = var.copy()
            
            # 根据视觉元素添加信息
            if 'bars' in visual_elements and var.get('role') == 'Y':
                enhanced_var['visualization_type'] = 'bar'
            elif 'circles' in visual_elements:
                enhanced_var['visualization_type'] = 'scatter'
            
            enhanced_vars.append(enhanced_var)
        
        return enhanced_vars


async def test_enhanced_annotator():
    """测试增强注释器"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        logger.error("需要设置MISTRAL_API_KEY")
        return
    
    annotator = EnhancedBBoxAnnotator(api_key, use_ocr=True)
    
    # 测试图表
    test_figure = DetectedFigure(
        page_index=0,
        bbox=BBox(x1=100, y1=200, x2=400, y2=500),
        figure_type=FigureType.FIGURE,
        caption="Test figure",
        confidence=0.95
    )
    
    pdf_path = Path("/path/to/test.pdf")
    
    try:
        annotation = await annotator.annotate_element(test_figure, pdf_path)
        print(json.dumps(annotation, indent=2))
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_annotator())