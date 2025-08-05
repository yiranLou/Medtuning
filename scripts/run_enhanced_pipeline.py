#!/usr/bin/env python3
"""
增强的医学文献多模态数据集构建主流水线
使用增强的图表和表格检测器
"""
import asyncio
import click
import logging
import sys
from pathlib import Path
from typing import Optional, List
import yaml
import json
from tqdm import tqdm
import os

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.core.schemas import DocumentAnnotation, BBoxAnnotation
from src.core.pdf_processor import PDFRenderer, RenderConfig
from src.core.pdf_processor.enhanced_detector import EnhancedFigureTableDetector
from src.annotation import (
    MistralConfig, 
    MistralClient,
    DocumentAnnotator,
    BBoxAnnotator
)
from src.dataset import (
    InternVL2Builder,
    DatasetSampler,
    MetaConfig,
    TaskType
)
from src.quality import ConsistencyChecker, DatasetDeduplicator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedPipeline:
    """增强的数据集构建流水线"""
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.setup_paths()
        
        # 设置Mistral API key
        if 'MISTRAL_API_KEY' not in os.environ:
            api_key = click.prompt('请输入Mistral API Key', hide_input=True)
            os.environ['MISTRAL_API_KEY'] = api_key
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_paths(self):
        """设置路径"""
        self.data_root = Path(self.config['paths']['data_root'])
        self.raw_pdfs_dir = Path(self.config['paths']['raw_pdfs'])
        self.processed_dir = Path(self.config['paths']['processed'])
        self.outputs_dir = Path(self.config['paths']['outputs'])
        
        # 创建必要的目录
        for path in [self.processed_dir, self.outputs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def run(
        self,
        pdf_files: Optional[List[Path]] = None,
        skip_detection: bool = False,
        skip_annotation: bool = False,
        skip_dataset: bool = False
    ):
        """运行完整流水线"""
        logger.info("=== 开始运行增强的数据集构建流水线 ===")
        
        # 1. 获取PDF文件列表
        if pdf_files is None:
            # 使用用户实际的PDF文件
            actual_pdf_dir = Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data")
            if actual_pdf_dir.exists():
                pdf_files = list(actual_pdf_dir.glob("*.pdf"))
                logger.info(f"使用实际PDF目录: {actual_pdf_dir}")
            else:
                pdf_files = list(self.raw_pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"未找到PDF文件")
            return
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 2. 处理每个PDF
        all_doc_annotations = []
        all_bbox_annotations = []
        
        for pdf_path in tqdm(pdf_files, desc="处理PDF"):
            try:
                doc_ann, bbox_anns = await self.process_single_pdf_enhanced(
                    pdf_path,
                    skip_detection=skip_detection,
                    skip_annotation=skip_annotation
                )
                
                if doc_ann:
                    all_doc_annotations.append(doc_ann)
                if bbox_anns:
                    all_bbox_annotations.extend(bbox_anns)
                    
            except Exception as e:
                logger.error(f"处理 {pdf_path} 失败: {e}")
                continue
        
        # 3. 质量控制
        logger.info("=== 执行质量控制 ===")
        all_doc_annotations, all_bbox_annotations = self.quality_control(
            all_doc_annotations,
            all_bbox_annotations
        )
        
        # 4. 生成数据集
        if not skip_dataset:
            logger.info("=== 生成InternVL2数据集 ===")
            await self.generate_dataset(all_doc_annotations, all_bbox_annotations)
        
        logger.info("=== 流水线完成 ===")
    
    async def process_single_pdf_enhanced(
        self,
        pdf_path: Path,
        skip_detection: bool = False,
        skip_annotation: bool = False
    ) -> tuple:
        """使用增强检测器处理单个PDF"""
        paper_id = pdf_path.stem
        paper_dir = self.processed_dir / paper_id
        paper_dir.mkdir(exist_ok=True)
        
        logger.info(f"处理: {pdf_path.name}")
        
        # 1. PDF渲染
        page_images_dir = paper_dir / "pages"
        page_images_dir.mkdir(exist_ok=True)
        
        render_config = RenderConfig(dpi=300)  # 使用更高的DPI
        
        with PDFRenderer(pdf_path, render_config) as renderer:
            # 渲染所有页面
            page_paths = renderer.render_all_pages(page_images_dir, prefix=f"{paper_id}_page")
            logger.info(f"渲染了 {len(page_paths)} 页")
        
        # 2. 增强的图表和表格检测
        detected_elements = {'figures': [], 'tables': [], 'equations': []}
        if not skip_detection:
            detector = EnhancedFigureTableDetector()
            detected_elements = detector.detect_all_elements(pdf_path)
            
            logger.info(f"增强检测结果:")
            logger.info(f"  - 图表: {len(detected_elements['figures'])}个")
            logger.info(f"  - 表格: {len(detected_elements['tables'])}个")
            logger.info(f"  - 公式: {len(detected_elements['equations'])}个")
            
            # 渲染检测到的元素
            figures_dir = paper_dir / "figures"
            tables_dir = paper_dir / "tables"
            equations_dir = paper_dir / "equations"
            
            for dir_path in [figures_dir, tables_dir, equations_dir]:
                dir_path.mkdir(exist_ok=True)
            
            with PDFRenderer(pdf_path, render_config) as renderer:
                # 保存图表
                for i, fig in enumerate(detected_elements['figures']):
                    output_path = figures_dir / f"figure_{i}.png"
                    try:
                        renderer.crop_region(fig.page_index, fig.bbox, output_path)
                        fig.crop_path = str(output_path.relative_to(self.processed_dir))
                    except Exception as e:
                        logger.warning(f"保存图表 {i} 失败: {e}")
                
                # 保存表格
                for i, table in enumerate(detected_elements['tables']):
                    output_path = tables_dir / f"table_{i}.png"
                    try:
                        renderer.crop_region(table.page_index, table.bbox, output_path)
                        table.crop_path = str(output_path.relative_to(self.processed_dir))
                    except Exception as e:
                        logger.warning(f"保存表格 {i} 失败: {e}")
                
                # 保存公式（只保存前10个）
                for i, eq in enumerate(detected_elements['equations'][:10]):
                    output_path = equations_dir / f"equation_{i}.png"
                    try:
                        renderer.crop_region(eq.page_index, eq.bbox, output_path)
                        eq.crop_path = str(output_path.relative_to(self.processed_dir))
                    except Exception as e:
                        logger.warning(f"保存公式 {i} 失败: {e}")
        
        if skip_annotation:
            return None, None
        
        # 3. 文档标注
        doc_annotator = DocumentAnnotator()
        doc_annotation = await doc_annotator.annotate_document(
            pdf_path,
            output_path=paper_dir / "document_annotation.json"
        )
        
        # 4. 边界框标注（包括图表和表格）
        bbox_annotations = []
        all_detected = (detected_elements['figures'] + 
                       detected_elements['tables'] + 
                       detected_elements['equations'][:5])  # 限制公式数量
        
        if all_detected:
            bbox_annotator = BBoxAnnotator()
            bbox_annotations = await bbox_annotator.annotate_figures(
                pdf_path,
                all_detected,
                output_dir=paper_dir,
                paper_id=paper_id
            )
        
        return doc_annotation, bbox_annotations
    
    def quality_control(
        self,
        doc_annotations: List[DocumentAnnotation],
        bbox_annotations: List[BBoxAnnotation]
    ) -> tuple:
        """质量控制"""
        # 1. 一致性检查
        checker = ConsistencyChecker(strict_mode=False)
        
        # 检查文档
        valid_docs = []
        for doc in doc_annotations:
            if checker.check_document_annotation(doc):
                valid_docs.append(doc)
            else:
                logger.warning(f"文档 {doc.paper_id} 未通过一致性检查")
        
        # 检查边界框
        valid_bboxes = []
        for bbox in bbox_annotations:
            # 这里需要页面尺寸，暂时使用默认值
            if checker.check_bbox_annotation(bbox, 2550, 3300):  # A4 at 300DPI
                valid_bboxes.append(bbox)
            else:
                logger.warning(f"边界框未通过一致性检查")
        
        # 2. 去重
        deduplicator = DatasetDeduplicator()
        unique_docs, unique_bboxes = deduplicator.deduplicate_dataset(
            valid_docs,
            valid_bboxes,
            self.processed_dir
        )
        
        logger.info(f"质量控制完成: 文档 {len(doc_annotations)} -> {len(unique_docs)}, "
                   f"边界框 {len(bbox_annotations)} -> {len(unique_bboxes)}")
        
        return unique_docs, unique_bboxes
    
    async def generate_dataset(
        self,
        doc_annotations: List[DocumentAnnotation],
        bbox_annotations: List[BBoxAnnotation]
    ):
        """生成InternVL2数据集"""
        # 准备图片映射
        page_images = {}
        crop_images = {}
        
        for doc in doc_annotations:
            paper_id = doc.paper_id
            paper_dir = self.processed_dir / paper_id / "pages"
            if paper_dir.exists():
                page_paths = sorted(paper_dir.glob("*.png"))
                page_images[paper_id] = [str(p.relative_to(self.processed_dir)) for p in page_paths]
        
        for bbox in bbox_annotations:
            key = f"{bbox.paper_id}_{bbox.page_index}_{bbox.bbox}"
            crop_images[key] = bbox.crop_path
        
        # 构建数据集
        builder = InternVL2Builder(image_base_path=self.processed_dir)
        
        # 使用增强的任务权重（更多表格相关任务）
        task_weights = {
            TaskType.PAGE_GROUNDING: 0.15,
            TaskType.FIGURE_CAPTION: 0.20,
            TaskType.VARIABLE_EXTRACTION: 0.15,
            TaskType.TABLE_READING: 0.25,  # 增加表格任务权重
            TaskType.MULTI_FIGURE: 0.15,
            TaskType.ABSTRACT_QA: 0.10
        }
        
        samples = builder.build_from_annotations(
            doc_annotations,
            bbox_annotations,
            page_images,
            crop_images,
            task_distribution=task_weights
        )
        
        # 采样
        sampler = DatasetSampler(task_weights=task_weights)
        target_size = min(50000, len(samples))  # 允许更大的数据集
        sampled = sampler.sample_dataset(
            [s.to_dict() for s in samples],
            target_size=target_size,
            random_seed=42
        )
        
        # 转换回样本对象
        final_samples = []
        for s_dict in sampled:
            # 从字典重建样本
            sample = InternVL2Builder._dict_to_sample(s_dict)
            if sample:
                final_samples.append(sample)
        
        # 保存数据集
        output_path = self.outputs_dir / "internvl2_enhanced_dataset.jsonl"
        builder.save_to_jsonl(final_samples, output_path)
        
        # 统计任务分布
        task_counts = {}
        for sample in final_samples:
            task_type = sample.conversations[0].get('value', '').split(':')[0]
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        logger.info("任务分布:")
        for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {task}: {count}")
        
        # 更新meta.json
        meta = MetaConfig()
        meta.add_data_source(
            "medical_papers_enhanced",
            str(output_path),
            repeat_time=1,
            metadata={
                "total_samples": len(final_samples),
                "papers": len(doc_annotations),
                "figures": sum(1 for b in bbox_annotations if b.figure_type == 'figure'),
                "tables": sum(1 for b in bbox_annotations if b.figure_type == 'table'),
                "equations": sum(1 for b in bbox_annotations if b.figure_type == 'equation'),
                "task_distribution": task_counts
            }
        )
        meta.save(self.outputs_dir / "meta_enhanced.json")
        
        logger.info(f"增强数据集生成完成: {len(final_samples)} 个样本")


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              default='configs/config.yaml', help='配置文件路径')
@click.option('--pdf-dir', '-d', type=click.Path(exists=True),
              help='PDF文件目录（覆盖配置）')
@click.option('--pdf-file', '-f', type=click.Path(exists=True),
              multiple=True, help='指定PDF文件')
@click.option('--skip-detection', is_flag=True, help='跳过图表检测')
@click.option('--skip-annotation', is_flag=True, help='跳过标注')
@click.option('--skip-dataset', is_flag=True, help='跳过数据集生成')
@click.option('--debug', is_flag=True, help='调试模式')
def main(config, pdf_dir, pdf_file, skip_detection, skip_annotation, skip_dataset, debug):
    """增强的医学文献多模态数据集构建工具"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 初始化流水线
    pipeline = EnhancedPipeline(config)
    
    # 准备PDF文件列表
    pdf_files = None
    if pdf_file:
        pdf_files = [Path(f) for f in pdf_file]
    elif pdf_dir:
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    # 运行流水线
    asyncio.run(pipeline.run(
        pdf_files=pdf_files,
        skip_detection=skip_detection,
        skip_annotation=skip_annotation,
        skip_dataset=skip_dataset
    ))


if __name__ == '__main__':
    main()