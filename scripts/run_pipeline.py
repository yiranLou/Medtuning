#!/usr/bin/env python3
"""
医学文献多模态数据集构建主流水线
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

from src.core.schemas import DocumentAnnotation, BBoxAnnotation, save_schemas_to_config
from src.core.pdf_processor import PDFRenderer, RenderConfig, create_detector
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


class Pipeline:
    """数据集构建流水线"""
    
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
        logger.info("=== 开始运行数据集构建流水线 ===")
        
        # 1. 获取PDF文件列表
        if pdf_files is None:
            pdf_files = list(self.raw_pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"未找到PDF文件在: {self.raw_pdfs_dir}")
            return
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 2. 处理每个PDF
        all_doc_annotations = []
        all_bbox_annotations = []
        
        for pdf_path in tqdm(pdf_files, desc="处理PDF"):
            try:
                doc_ann, bbox_anns = await self.process_single_pdf(
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
    
    async def process_single_pdf(
        self,
        pdf_path: Path,
        skip_detection: bool = False,
        skip_annotation: bool = False
    ) -> tuple:
        """处理单个PDF"""
        paper_id = pdf_path.stem
        paper_dir = self.processed_dir / paper_id
        paper_dir.mkdir(exist_ok=True)
        
        logger.info(f"处理: {pdf_path.name}")
        
        # 1. PDF渲染
        page_images_dir = paper_dir / "pages"
        page_images_dir.mkdir(exist_ok=True)
        
        render_config = RenderConfig(**self.config['pdf_processing']['renderer'])
        
        with PDFRenderer(pdf_path, render_config) as renderer:
            # 渲染所有页面
            page_paths = renderer.render_all_pages(page_images_dir, prefix=f"{paper_id}_page")
            logger.info(f"渲染了 {len(page_paths)} 页")
        
        # 2. 图表检测
        detected_figures = []
        if not skip_detection:
            detector = create_detector(use_pdffigures2=self.config['pdf_processing']['detector']['use_pdffigures2'])
            detected_figures = detector.detect(pdf_path)
            logger.info(f"检测到 {len(detected_figures)} 个图表")
        
        if skip_annotation:
            return None, None
        
        # 3. 文档标注
        doc_annotator = DocumentAnnotator()
        doc_annotation = await doc_annotator.annotate_document(
            pdf_path,
            output_path=paper_dir / "document_annotation.json"
        )
        
        # 4. 边界框标注
        bbox_annotations = []
        if detected_figures:
            bbox_annotator = BBoxAnnotator()
            bbox_annotations = await bbox_annotator.annotate_figures(
                pdf_path,
                detected_figures,
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
        checker = ConsistencyChecker(strict_mode=self.config['quality_control']['consistency']['strict_mode'])
        
        # 检查文档
        valid_docs = []
        for doc in doc_annotations:
            if checker.check_document_annotation(doc):
                valid_docs.append(doc)
            else:
                logger.warning(f"文档 {doc.paper_id} 未通过一致性检查:\n{checker.generate_report()}")
        
        # 检查边界框
        valid_bboxes = []
        for bbox in bbox_annotations:
            # 这里需要页面尺寸，暂时使用默认值
            if checker.check_bbox_annotation(bbox, 2000, 2800):
                valid_bboxes.append(bbox)
            else:
                logger.warning(f"边界框未通过一致性检查:\n{checker.generate_report()}")
        
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
        
        # 根据配置生成样本
        task_weights = {
            TaskType(k): v 
            for k, v in self.config['internvl2']['task_weights'].items()
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
        target_size = min(10000, len(samples))  # 限制数据集大小
        sampled = sampler.sample_dataset(
            [s.to_dict() for s in samples],
            target_size=target_size,
            random_seed=self.config['internvl2']['sampling']['random_seed']
        )
        
        # 转换回样本对象
        final_samples = []
        for s_dict in sampled:
            # 从字典重建样本
            sample = InternVL2Builder._dict_to_sample(s_dict)
            if sample:
                final_samples.append(sample)
        
        # 保存数据集
        output_path = self.outputs_dir / "internvl2_dataset.jsonl"
        builder.save_to_jsonl(final_samples, output_path)
        
        # 更新meta.json
        meta = MetaConfig()
        meta.add_data_source(
            "medical_papers",
            str(output_path),
            repeat_time=1,
            metadata={
                "total_samples": len(final_samples),
                "papers": len(doc_annotations),
                "figures": len(bbox_annotations)
            }
        )
        meta.save(self.outputs_dir / "meta.json")
        
        logger.info(f"数据集生成完成: {len(final_samples)} 个样本")


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
    """医学文献多模态数据集构建工具"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 初始化流水线
    pipeline = Pipeline(config)
    
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