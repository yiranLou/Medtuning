"""PDF渲染器，将PDF转换为高质量图片"""
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import logging

from ..schemas import BBox

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """渲染配置"""
    page_dpi: int = 200  # 页面渲染DPI（A4@200DPI ≈ 1654×2339）
    crop_dpi: int = 300  # 裁剪图渲染DPI（更高清晰度）
    color_mode: str = "RGB"  # 颜色模式
    image_format: str = "PNG"  # 输出格式
    expand_margin: int = 16  # 裁剪扩边像素
    max_dimension: int = 4096  # 最大维度限制（防止爆显存）
    jpeg_quality: int = 95  # JPEG质量（如果使用）


class PDFRenderer:
    """PDF渲染器"""
    
    def __init__(self, pdf_path: Union[str, Path], config: Optional[RenderConfig] = None):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {self.pdf_path}")
        
        self.config = config or RenderConfig()
        self.doc = fitz.open(str(self.pdf_path))
        self.page_count = len(self.doc)
        
        # 缓存页面尺寸
        self._page_sizes = self._get_page_sizes()
    
    def _get_page_sizes(self) -> List[Tuple[float, float]]:
        """获取所有页面的原始尺寸"""
        sizes = []
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            rect = page.rect
            sizes.append((rect.width, rect.height))
        return sizes
    
    def get_page_size_at_dpi(self, page_index: int, dpi: int) -> Tuple[int, int]:
        """获取指定DPI下的页面尺寸"""
        if page_index < 0 or page_index >= self.page_count:
            raise ValueError(f"页面索引超出范围: {page_index}")
        
        width, height = self._page_sizes[page_index]
        scale = dpi / 72.0  # PDF标准是72 DPI
        
        pixel_width = int(width * scale)
        pixel_height = int(height * scale)
        
        # 检查尺寸限制
        if max(pixel_width, pixel_height) > self.config.max_dimension:
            # 按比例缩小
            ratio = self.config.max_dimension / max(pixel_width, pixel_height)
            pixel_width = int(pixel_width * ratio)
            pixel_height = int(pixel_height * ratio)
            logger.warning(f"页面{page_index}尺寸过大，已缩放至{pixel_width}x{pixel_height}")
        
        return pixel_width, pixel_height
    
    def render_page(self, page_index: int, output_path: Optional[Path] = None) -> Image.Image:
        """渲染单个页面为图片"""
        if page_index < 0 or page_index >= self.page_count:
            raise ValueError(f"页面索引超出范围: {page_index}")
        
        page = self.doc[page_index]
        
        # 计算缩放因子
        mat = fitz.Matrix(self.config.page_dpi / 72.0, self.config.page_dpi / 72.0)
        
        # 渲染页面
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 转换为PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 确保是RGB模式
        if img.mode != self.config.color_mode:
            img = img.convert(self.config.color_mode)
        
        # 保存到文件（如果指定）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.image_format.upper() == "JPEG":
                img.save(output_path, "JPEG", quality=self.config.jpeg_quality)
            else:
                img.save(output_path, self.config.image_format)
            
            logger.info(f"已保存页面{page_index}到: {output_path}")
        
        return img
    
    def render_all_pages(self, output_dir: Path, prefix: str = "page") -> List[Path]:
        """渲染所有页面"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        for page_idx in range(self.page_count):
            filename = f"{prefix}_{page_idx:03d}.{self.config.image_format.lower()}"
            output_path = output_dir / filename
            self.render_page(page_idx, output_path)
            output_paths.append(output_path)
        
        logger.info(f"已渲染{self.page_count}个页面到: {output_dir}")
        return output_paths
    
    def crop_region(
        self, 
        page_index: int, 
        bbox: Union[BBox, List[int], Tuple[int, int, int, int]],
        output_path: Optional[Path] = None,
        use_high_dpi: bool = True
    ) -> Image.Image:
        """裁剪页面的指定区域"""
        if page_index < 0 or page_index >= self.page_count:
            raise ValueError(f"页面索引超出范围: {page_index}")
        
        # 转换bbox格式
        if isinstance(bbox, (list, tuple)):
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        
        page = self.doc[page_index]
        
        # 使用更高的DPI进行裁剪
        dpi = self.config.crop_dpi if use_high_dpi else self.config.page_dpi
        scale = dpi / 72.0
        
        # 将像素坐标转换回PDF坐标
        pdf_x1 = x1 / (self.config.page_dpi / 72.0)
        pdf_y1 = y1 / (self.config.page_dpi / 72.0)
        pdf_x2 = x2 / (self.config.page_dpi / 72.0)
        pdf_y2 = y2 / (self.config.page_dpi / 72.0)
        
        # 添加扩边
        margin = self.config.expand_margin / (self.config.page_dpi / 72.0)
        pdf_x1 = max(0, pdf_x1 - margin)
        pdf_y1 = max(0, pdf_y1 - margin)
        pdf_x2 = min(page.rect.width, pdf_x2 + margin)
        pdf_y2 = min(page.rect.height, pdf_y2 + margin)
        
        # 创建裁剪区域
        clip_rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)
        
        # 渲染裁剪区域
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        
        # 转换为PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 确保是RGB模式
        if img.mode != self.config.color_mode:
            img = img.convert(self.config.color_mode)
        
        # 保存到文件（如果指定）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.image_format.upper() == "JPEG":
                img.save(output_path, "JPEG", quality=self.config.jpeg_quality)
            else:
                img.save(output_path, self.config.image_format)
            
            logger.info(f"已保存裁剪图到: {output_path}")
        
        return img
    
    def extract_text_in_bbox(
        self, 
        page_index: int, 
        bbox: Union[BBox, List[int], Tuple[int, int, int, int]]
    ) -> str:
        """提取边界框内的文本（用于锚定）"""
        if page_index < 0 or page_index >= self.page_count:
            raise ValueError(f"页面索引超出范围: {page_index}")
        
        # 转换bbox格式
        if isinstance(bbox, (list, tuple)):
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        
        page = self.doc[page_index]
        
        # 将像素坐标转换回PDF坐标
        scale = self.config.page_dpi / 72.0
        pdf_rect = fitz.Rect(
            x1 / scale,
            y1 / scale,
            x2 / scale,
            y2 / scale
        )
        
        # 提取文本
        text = page.get_text("text", clip=pdf_rect)
        return text.strip()
    
    def close(self):
        """关闭PDF文档"""
        if self.doc:
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()