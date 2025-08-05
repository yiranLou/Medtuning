"""InternVL2 JSONL数据集生成器"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from PIL import Image
import random

from .qa_templates import QAGenerator, TaskType
from ..core.schemas import DocumentAnnotation, BBoxAnnotation

logger = logging.getLogger(__name__)


@dataclass
class InternVL2Sample:
    """InternVL2训练样本"""
    id: str
    image: Union[str, List[str]]  # 图片路径（单图或多图）
    conversations: List[Dict[str, str]]  # 对话历史
    width: Union[int, List[int]]  # 图片宽度
    height: Union[int, List[int]]  # 图片高度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            "id": self.id,
            "image": self.image,
            "conversations": self.conversations
        }
        
        # 单图
        if isinstance(self.image, str):
            data["width"] = self.width
            data["height"] = self.height
        # 多图
        else:
            data["width_list"] = self.width
            data["height_list"] = self.height
        
        return data
    
    def validate(self) -> bool:
        """验证样本合法性"""
        # 检查图片数量
        if isinstance(self.image, list):
            image_count = len(self.image)
            # 检查宽高列表
            if not isinstance(self.width, list) or len(self.width) != image_count:
                logger.error(f"样本{self.id}: width_list长度与图片数量不匹配")
                return False
            if not isinstance(self.height, list) or len(self.height) != image_count:
                logger.error(f"样本{self.id}: height_list长度与图片数量不匹配")
                return False
        
        # 检查对话中的<image>标记
        full_text = " ".join([conv["value"] for conv in self.conversations])
        image_tag_count = full_text.count("<image>")
        expected_count = len(self.image) if isinstance(self.image, list) else 1
        
        if image_tag_count != expected_count:
            logger.error(
                f"样本{self.id}: <image>标记数量({image_tag_count})与图片数量({expected_count})不匹配"
            )
            return False
        
        # 检查坐标
        if "<box>" in full_text:
            # TODO: 验证坐标是否在图片范围内
            pass
        
        return True


class InternVL2Builder:
    """InternVL2数据集构建器"""
    
    def __init__(
        self,
        qa_generator: Optional[QAGenerator] = None,
        image_base_path: Optional[Path] = None
    ):
        self.qa_generator = qa_generator or QAGenerator()
        self.image_base_path = Path(image_base_path) if image_base_path else Path(".")
    
    def build_page_grounding_sample(
        self,
        doc_annotation: DocumentAnnotation,
        page_image_path: str,
        page_index: int,
        section_bboxes: List[Dict[str, Any]]
    ) -> InternVL2Sample:
        """构建页面定位样本"""
        # 随机选择一个章节
        section_bbox = random.choice(section_bboxes)
        
        # 生成问题
        question = f"请在页面中找到「{section_bbox['title']}」这一节的标题，并标出其位置。"
        
        # 生成答案（包含grounding）
        answer = f"<ref>{section_bbox['title']}</ref><box>{section_bbox['bbox']}</box>"
        
        # 获取图片尺寸
        image_path = self.image_base_path / page_image_path
        width, height = self._get_image_size(image_path)
        
        # 构建对话
        conversations = [
            {
                "from": "human",
                "value": f"<image>\n{question}"
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
        
        return InternVL2Sample(
            id=f"{doc_annotation.paper_id}_p{page_index}_grounding",
            image=page_image_path,
            conversations=conversations,
            width=width,
            height=height
        )
    
    def build_figure_caption_sample(
        self,
        bbox_annotation: BBoxAnnotation,
        crop_image_path: str
    ) -> InternVL2Sample:
        """构建图表摘要样本"""
        # 生成Q/A
        qa_pairs = self.qa_generator.generate_for_bbox(
            bbox_annotation,
            [TaskType.FIGURE_CAPTION]
        )
        
        if not qa_pairs:
            return None
        
        qa = qa_pairs[0]
        
        # 获取图片尺寸
        image_path = self.image_base_path / crop_image_path
        width, height = self._get_image_size(image_path)
        
        # 构建对话
        conversations = [
            {
                "from": "human",
                "value": f"<image>\n{qa['question']}"
            },
            {
                "from": "gpt",
                "value": qa['answer']
            }
        ]
        
        return InternVL2Sample(
            id=f"{bbox_annotation.paper_id}_p{bbox_annotation.page_index}_fig_caption",
            image=crop_image_path,
            conversations=conversations,
            width=width,
            height=height
        )
    
    def build_multi_figure_sample(
        self,
        bbox_annotations: List[BBoxAnnotation],
        crop_image_paths: List[str]
    ) -> InternVL2Sample:
        """构建多图对比样本"""
        if len(bbox_annotations) < 2:
            return None
        
        # 生成问题
        question = f"比较这{len(bbox_annotations)}个图表，它们展示了什么趋势或区别？"
        
        # 生成答案
        answer_parts = []
        for i, ann in enumerate(bbox_annotations):
            if ann.key_findings:
                answer_parts.append(f"图{i+1}：{ann.key_findings}")
            elif ann.caption:
                answer_parts.append(f"图{i+1}：{ann.caption[:50]}...")
        
        answer = "\n".join(answer_parts)
        
        # 获取所有图片尺寸
        widths = []
        heights = []
        for path in crop_image_paths:
            image_path = self.image_base_path / path
            w, h = self._get_image_size(image_path)
            widths.append(w)
            heights.append(h)
        
        # 构建对话（多个<image>标记）
        image_tags = "\n".join(["<image>"] * len(crop_image_paths))
        conversations = [
            {
                "from": "human",
                "value": f"{image_tags}\n{question}"
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
        
        return InternVL2Sample(
            id=f"{bbox_annotations[0].paper_id}_multi_fig",
            image=crop_image_paths,
            conversations=conversations,
            width=widths,
            height=heights
        )
    
    def build_table_reading_sample(
        self,
        bbox_annotation: BBoxAnnotation,
        crop_image_path: str
    ) -> InternVL2Sample:
        """构建表格读取样本"""
        if bbox_annotation.figure_type != "table" or not bbox_annotation.table_csv:
            return None
        
        # 生成Q/A
        qa_pairs = self.qa_generator.generate_for_bbox(
            bbox_annotation,
            [TaskType.TABLE_READING]
        )
        
        if not qa_pairs:
            return None
        
        qa = qa_pairs[0]
        
        # 获取图片尺寸
        image_path = self.image_base_path / crop_image_path
        width, height = self._get_image_size(image_path)
        
        # 构建对话
        conversations = [
            {
                "from": "human",
                "value": f"<image>\n{qa['question']}"
            },
            {
                "from": "gpt",
                "value": qa['answer']
            }
        ]
        
        return InternVL2Sample(
            id=f"{bbox_annotation.paper_id}_p{bbox_annotation.page_index}_table",
            image=crop_image_path,
            conversations=conversations,
            width=width,
            height=height
        )
    
    def build_abstract_qa_sample(
        self,
        doc_annotation: DocumentAnnotation,
        page_image_path: str
    ) -> InternVL2Sample:
        """构建摘要问答样本（使用第一页）"""
        # 生成Q/A
        qa_pairs = self.qa_generator.generate_for_document(
            doc_annotation,
            [TaskType.ABSTRACT_QA]
        )
        
        if not qa_pairs:
            return None
        
        qa = qa_pairs[0]
        
        # 获取图片尺寸
        image_path = self.image_base_path / page_image_path
        width, height = self._get_image_size(image_path)
        
        # 构建对话
        conversations = [
            {
                "from": "human",
                "value": f"<image>\n{qa['question']}"
            },
            {
                "from": "gpt",
                "value": qa['answer']
            }
        ]
        
        return InternVL2Sample(
            id=f"{doc_annotation.paper_id}_abstract_qa",
            image=page_image_path,
            conversations=conversations,
            width=width,
            height=height
        )
    
    def build_from_annotations(
        self,
        doc_annotations: List[DocumentAnnotation],
        bbox_annotations: List[BBoxAnnotation],
        page_images: Dict[str, List[str]],  # paper_id -> [page_paths]
        crop_images: Dict[str, str],  # bbox_id -> crop_path
        task_distribution: Dict[TaskType, float] = None
    ) -> List[InternVL2Sample]:
        """从标注构建数据集"""
        if task_distribution is None:
            # 默认分布
            task_distribution = {
                TaskType.PAGE_GROUNDING: 0.2,
                TaskType.FIGURE_CAPTION: 0.5,
                TaskType.VARIABLE_EXTRACTION: 0.1,
                TaskType.TABLE_READING: 0.1,
                TaskType.MULTI_FIGURE: 0.05,
                TaskType.ABSTRACT_QA: 0.05
            }
        
        samples = []
        
        # 1. 生成页面定位样本
        # TODO: 需要章节的bbox信息
        
        # 2. 生成图表摘要样本
        for bbox_ann in bbox_annotations:
            crop_path = crop_images.get(f"{bbox_ann.paper_id}_{bbox_ann.page_index}_{bbox_ann.bbox}")
            if crop_path:
                sample = self.build_figure_caption_sample(bbox_ann, crop_path)
                if sample and sample.validate():
                    samples.append(sample)
        
        # 3. 生成表格读取样本
        table_annotations = [
            ann for ann in bbox_annotations 
            if ann.figure_type == "table" and ann.table_csv
        ]
        for table_ann in table_annotations:
            crop_path = crop_images.get(f"{table_ann.paper_id}_{table_ann.page_index}_{table_ann.bbox}")
            if crop_path:
                sample = self.build_table_reading_sample(table_ann, crop_path)
                if sample and sample.validate():
                    samples.append(sample)
        
        # 4. 生成多图对比样本（同一篇论文的图表）
        papers_bbox = {}
        for bbox_ann in bbox_annotations:
            if bbox_ann.paper_id not in papers_bbox:
                papers_bbox[bbox_ann.paper_id] = []
            papers_bbox[bbox_ann.paper_id].append(bbox_ann)
        
        for paper_id, paper_bboxes in papers_bbox.items():
            if len(paper_bboxes) >= 2:
                # 随机选择2-3个图表
                selected = random.sample(paper_bboxes, min(3, len(paper_bboxes)))
                paths = []
                for bbox_ann in selected:
                    crop_path = crop_images.get(
                        f"{bbox_ann.paper_id}_{bbox_ann.page_index}_{bbox_ann.bbox}"
                    )
                    if crop_path:
                        paths.append(crop_path)
                
                if len(paths) >= 2:
                    sample = self.build_multi_figure_sample(selected[:len(paths)], paths)
                    if sample and sample.validate():
                        samples.append(sample)
        
        # 5. 生成摘要问答样本
        for doc_ann in doc_annotations:
            if doc_ann.paper_id in page_images and page_images[doc_ann.paper_id]:
                # 使用第一页
                first_page = page_images[doc_ann.paper_id][0]
                sample = self.build_abstract_qa_sample(doc_ann, first_page)
                if sample and sample.validate():
                    samples.append(sample)
        
        logger.info(f"共生成{len(samples)}个训练样本")
        return samples
    
    def save_to_jsonl(
        self,
        samples: List[InternVL2Sample],
        output_path: Path,
        validate: bool = True
    ):
        """保存为JSONL格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                if validate and not sample.validate():
                    logger.warning(f"跳过无效样本: {sample.id}")
                    continue
                
                # 转换为字典并写入
                data = sample.to_dict()
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                valid_count += 1
        
        logger.info(f"已保存{valid_count}个有效样本到: {output_path}")
    
    def _get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """获取图片尺寸"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.error(f"无法读取图片{image_path}: {e}")
            # 返回默认尺寸
            return 1024, 1024