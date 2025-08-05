"""边界框级标注器"""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import json
from PIL import Image

from .mistral_client import MistralClient
from ..core.schemas import BBoxAnnotation, BBoxPage, BBox
from ..core.pdf_processor import PDFRenderer, DetectedFigure

logger = logging.getLogger(__name__)


class BBoxAnnotator:
    """边界框级标注器"""
    
    def __init__(
        self,
        mistral_client: Optional[MistralClient] = None,
        use_anchor_text: bool = True,
        max_concurrent: int = 5
    ):
        self.client = mistral_client or MistralClient()
        self.use_anchor_text = use_anchor_text
        self.max_concurrent = max_concurrent
    
    async def annotate_figures(
        self,
        pdf_path: Path,
        detected_figures: List[DetectedFigure],
        output_dir: Optional[Path] = None,
        paper_id: Optional[str] = None
    ) -> List[BBoxAnnotation]:
        """标注检测到的图表"""
        pdf_path = Path(pdf_path)
        
        if not paper_id:
            paper_id = pdf_path.stem
        
        # 准备输出目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            crop_dir = output_dir / "crops"
            crop_dir.mkdir(exist_ok=True)
        else:
            crop_dir = None
        
        # 渲染PDF
        with PDFRenderer(pdf_path) as renderer:
            # 按页分组
            figures_by_page = self._group_by_page(detected_figures)
            
            # 准备标注任务
            tasks = []
            for page_idx, page_figures in figures_by_page.items():
                # 渲染页面
                page_image = renderer.render_page(page_idx)
                page_width, page_height = page_image.size
                
                for i, figure in enumerate(page_figures):
                    # 裁剪图表
                    crop_image = renderer.crop_region(page_idx, figure.bbox)
                    
                    # 保存裁剪图
                    if crop_dir:
                        crop_filename = f"{paper_id}_p{page_idx:03d}_fig{i:02d}.png"
                        crop_path = crop_dir / crop_filename
                        crop_image.save(crop_path)
                        relative_crop_path = f"crops/{crop_filename}"
                    else:
                        relative_crop_path = f"temp_crop_{page_idx}_{i}.png"
                    
                    # 提取锚定文本
                    anchor_text = None
                    if self.use_anchor_text:
                        anchor_text = renderer.extract_text_in_bbox(
                            page_idx, 
                            self._expand_bbox(figure.bbox, 50)
                        )
                    
                    # 创建任务
                    task = {
                        'type': 'bbox',
                        'params': {
                            'crop_image': crop_image,
                            'page_image': page_image if self.use_anchor_text else None,
                            'bbox_coords': figure.bbox.to_list(),
                            'anchor_text': anchor_text,
                            'additional_instructions': self._build_figure_instructions(figure)
                        },
                        'metadata': {
                            'paper_id': paper_id,
                            'page_index': page_idx,
                            'page_width': page_width,
                            'page_height': page_height,
                            'bbox': figure.bbox,
                            'crop_path': relative_crop_path,
                            'figure_type': figure.figure_type,
                            'caption': figure.caption
                        }
                    }
                    tasks.append(task)
        
        # 批量标注
        results = await self._annotate_batch(tasks)
        
        # 后处理结果
        annotations = self._postprocess_results(results, tasks)
        
        # 保存结果
        if output_dir:
            self._save_annotations(annotations, output_dir / "bbox_annotations.json")
        
        return annotations
    
    def _group_by_page(self, figures: List[DetectedFigure]) -> Dict[int, List[DetectedFigure]]:
        """按页分组图表"""
        grouped = {}
        for fig in figures:
            if fig.page_index not in grouped:
                grouped[fig.page_index] = []
            grouped[fig.page_index].append(fig)
        
        # 按位置排序
        for page_figures in grouped.values():
            page_figures.sort(key=lambda f: (f.bbox.y1, f.bbox.x1))
        
        return grouped
    
    def _expand_bbox(self, bbox: BBox, margin: int) -> BBox:
        """扩展边界框"""
        return BBox(
            x1=max(0, bbox.x1 - margin),
            y1=max(0, bbox.y1 - margin),
            x2=bbox.x2 + margin,
            y2=bbox.y2 + margin
        )
    
    def _build_figure_instructions(self, figure: DetectedFigure) -> str:
        """构建图表特定的指令"""
        instructions = []
        
        if figure.figure_type == "table":
            instructions.append("这是一个表格，请尽可能提取table_csv字段")
        elif figure.figure_type == "figure":
            instructions.append("这是一个图表，请提取坐标轴信息和变量")
        elif figure.figure_type == "equation":
            instructions.append("这是一个公式，请在caption中包含LaTeX格式")
        
        if figure.caption:
            instructions.append(f"参考标题: {figure.caption}")
        
        return "\n".join(instructions)
    
    async def _annotate_batch(self, tasks: List[Dict]) -> List[Any]:
        """批量标注"""
        results = await self.client.annotate_batch(
            tasks, 
            max_concurrent=self.max_concurrent
        )
        return results
    
    def _postprocess_results(
        self, 
        results: List[Any], 
        tasks: List[Dict]
    ) -> List[BBoxAnnotation]:
        """后处理标注结果"""
        annotations = []
        
        for result, task in zip(results, tasks):
            if isinstance(result, Exception):
                logger.error(f"标注失败: {result}")
                # 创建基础标注
                annotation = self._create_fallback_annotation(task['metadata'])
            else:
                # 合并元数据
                annotation = result
                metadata = task['metadata']
                
                # 更新字段
                annotation.paper_id = metadata['paper_id']
                annotation.page_index = metadata['page_index']
                annotation.bbox = metadata['bbox']
                annotation.crop_path = metadata['crop_path']
                
                # 如果API没有返回figure_type，使用检测到的类型
                if not annotation.figure_type:
                    annotation.figure_type = metadata['figure_type']
                
                # 如果有原始caption但API没有返回，使用原始的
                if metadata.get('caption') and not annotation.caption:
                    annotation.caption = metadata['caption']
                
                # 验证坐标
                try:
                    annotation.validate_bbox_within_page(
                        metadata['page_width'],
                        metadata['page_height']
                    )
                except ValueError as e:
                    logger.warning(f"坐标验证失败: {e}")
                    # 修正坐标
                    annotation.bbox = self._fix_bbox_coords(
                        annotation.bbox,
                        metadata['page_width'],
                        metadata['page_height']
                    )
            
            annotations.append(annotation)
        
        return annotations
    
    def _create_fallback_annotation(self, metadata: Dict) -> BBoxAnnotation:
        """创建降级标注"""
        return BBoxAnnotation(
            paper_id=metadata['paper_id'],
            page_index=metadata['page_index'],
            bbox=metadata['bbox'],
            crop_path=metadata['crop_path'],
            figure_type=metadata.get('figure_type', 'other'),
            caption=metadata.get('caption'),
            confidence_score=0.0  # 表示这是降级结果
        )
    
    def _fix_bbox_coords(
        self, 
        bbox: BBox, 
        page_width: int, 
        page_height: int
    ) -> BBox:
        """修正超出页面的坐标"""
        return BBox(
            x1=max(0, min(bbox.x1, page_width - 1)),
            y1=max(0, min(bbox.y1, page_height - 1)),
            x2=max(1, min(bbox.x2, page_width)),
            y2=max(1, min(bbox.y2, page_height))
        )
    
    def _save_annotations(self, annotations: List[BBoxAnnotation], output_path: Path):
        """保存标注结果"""
        # 按页分组
        pages = {}
        for ann in annotations:
            page_key = (ann.paper_id, ann.page_index)
            if page_key not in pages:
                pages[page_key] = []
            pages[page_key].append(ann)
        
        # 构建输出数据
        output_data = []
        for (paper_id, page_index), page_annotations in pages.items():
            # 获取页面尺寸（从第一个标注推断）
            if page_annotations:
                first_ann = page_annotations[0]
                # 这里需要实际的页面尺寸，暂时使用估计值
                page_data = BBoxPage(
                    paper_id=paper_id,
                    page_index=page_index,
                    page_width=2000,  # TODO: 获取实际尺寸
                    page_height=2800,
                    annotations=page_annotations
                )
                output_data.append(page_data.model_dump(exclude_none=True))
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存{len(annotations)}个边界框标注到: {output_path}")


class TableExtractor:
    """表格数据提取器"""
    
    def extract_table_csv(
        self,
        image: Image.Image,
        bbox_annotation: BBoxAnnotation
    ) -> Optional[str]:
        """从图像中提取表格CSV"""
        # TODO: 实现表格提取逻辑
        # 可以使用:
        # 1. OCR + 表格识别
        # 2. 调用专门的表格提取API
        # 3. 使用开源表格提取工具
        return None