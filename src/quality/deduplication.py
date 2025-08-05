"""数据去重模块"""
import hashlib
from typing import List, Dict, Any, Tuple, Set, Optional
from pathlib import Path
import json
import logging
from collections import defaultdict
import numpy as np
from PIL import Image
import imagehash

from ..core.schemas import BBoxAnnotation, DocumentAnnotation

logger = logging.getLogger(__name__)


class TextDeduplicator:
    """文本去重器"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        min_length: int = 50
    ):
        self.similarity_threshold = similarity_threshold
        self.min_length = min_length
    
    def deduplicate_documents(
        self,
        documents: List[DocumentAnnotation]
    ) -> List[DocumentAnnotation]:
        """去重文档标注"""
        if not documents:
            return []
        
        # 1. 基于paper_id去重
        unique_by_id = {}
        for doc in documents:
            if doc.paper_id not in unique_by_id:
                unique_by_id[doc.paper_id] = doc
            else:
                # 选择信息更完整的版本
                existing = unique_by_id[doc.paper_id]
                if self._is_more_complete(doc, existing):
                    unique_by_id[doc.paper_id] = doc
        
        documents = list(unique_by_id.values())
        
        # 2. 基于标题去重
        unique_docs = []
        seen_titles = set()
        
        for doc in documents:
            title_key = self._normalize_title(doc.title)
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_docs.append(doc)
            else:
                logger.warning(f"发现重复标题: {doc.title}")
        
        # 3. 基于摘要相似度去重
        final_docs = []
        abstract_hashes = []
        
        for doc in unique_docs:
            if len(doc.abstract) < self.min_length:
                final_docs.append(doc)
                continue
            
            # 计算摘要hash
            abstract_hash = self._compute_text_hash(doc.abstract)
            
            # 检查是否与已有摘要相似
            is_duplicate = False
            for existing_hash in abstract_hashes:
                similarity = self._hash_similarity(abstract_hash, existing_hash)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    logger.warning(f"发现相似摘要: {doc.paper_id}")
                    break
            
            if not is_duplicate:
                abstract_hashes.append(abstract_hash)
                final_docs.append(doc)
        
        logger.info(
            f"文档去重: {len(documents)} -> {len(final_docs)} "
            f"(移除{len(documents) - len(final_docs)}个重复)"
        )
        
        return final_docs
    
    def _normalize_title(self, title: str) -> str:
        """标准化标题用于比较"""
        # 转小写，去除空白和标点
        title = title.lower().strip()
        title = ''.join(c for c in title if c.isalnum() or c.isspace())
        title = ' '.join(title.split())  # 规范化空白
        return title
    
    def _is_more_complete(self, doc1: DocumentAnnotation, doc2: DocumentAnnotation) -> bool:
        """判断doc1是否比doc2更完整"""
        score1 = 0
        score2 = 0
        
        # 比较各字段的完整性
        if doc1.abstract and len(doc1.abstract) > len(doc2.abstract or ''):
            score1 += 1
        else:
            score2 += 1
        
        if len(doc1.sections) > len(doc2.sections):
            score1 += 1
        else:
            score2 += 1
        
        if doc1.authors and len(doc1.authors) > len(doc2.authors or []):
            score1 += 1
        else:
            score2 += 1
        
        if doc1.references and len(doc1.references) > len(doc2.references or []):
            score1 += 1
        else:
            score2 += 1
        
        return score1 > score2
    
    def _compute_text_hash(self, text: str) -> np.ndarray:
        """计算文本的向量hash"""
        # 简单的词频向量
        words = text.lower().split()
        word_counts = defaultdict(int)
        
        for word in words:
            if len(word) > 2:  # 忽略短词
                word_counts[word] += 1
        
        # 选择最频繁的100个词
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
        
        # 创建hash向量
        hash_vector = []
        for word, count in top_words:
            # 使用词和计数创建特征
            feature = f"{word}:{count}"
            hash_value = int(hashlib.md5(feature.encode()).hexdigest()[:8], 16)
            hash_vector.append(hash_value)
        
        return np.array(hash_vector)
    
    def _hash_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """计算两个hash的相似度"""
        if len(hash1) == 0 or len(hash2) == 0:
            return 0.0
        
        # 计算Jaccard相似度
        set1 = set(hash1)
        set2 = set(hash2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class ImageDeduplicator:
    """图像去重器"""
    
    def __init__(
        self,
        hash_size: int = 16,
        max_distance: int = 5
    ):
        self.hash_size = hash_size
        self.max_distance = max_distance
    
    def deduplicate_bbox_annotations(
        self,
        annotations: List[BBoxAnnotation],
        image_dir: Path,
        iou_threshold: float = 0.9
    ) -> List[BBoxAnnotation]:
        """去重边界框标注"""
        if not annotations:
            return []
        
        # 1. 基于位置去重（同页面、高IoU）
        annotations = self._deduplicate_by_position(annotations, iou_threshold)
        
        # 2. 基于图像内容去重
        annotations = self._deduplicate_by_image_content(annotations, image_dir)
        
        return annotations
    
    def _deduplicate_by_position(
        self,
        annotations: List[BBoxAnnotation],
        iou_threshold: float
    ) -> List[BBoxAnnotation]:
        """基于位置去重"""
        # 按页面分组
        page_groups = defaultdict(list)
        for ann in annotations:
            key = (ann.paper_id, ann.page_index)
            page_groups[key].append(ann)
        
        unique_annotations = []
        
        for (paper_id, page_index), page_anns in page_groups.items():
            # 按面积降序排序
            page_anns.sort(
                key=lambda a: (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1),
                reverse=True
            )
            
            # 去重
            kept = []
            for ann in page_anns:
                is_duplicate = False
                
                for kept_ann in kept:
                    iou = self._compute_iou(ann.bbox, kept_ann.bbox)
                    if iou > iou_threshold:
                        # 检查是否是相同类型
                        if ann.figure_type == kept_ann.figure_type:
                            is_duplicate = True
                            logger.debug(
                                f"位置重复: {ann.paper_id} p{ann.page_index} "
                                f"IoU={iou:.2f}"
                            )
                            break
                
                if not is_duplicate:
                    kept.append(ann)
            
            unique_annotations.extend(kept)
        
        logger.info(
            f"位置去重: {len(annotations)} -> {len(unique_annotations)} "
            f"(移除{len(annotations) - len(unique_annotations)}个重复)"
        )
        
        return unique_annotations
    
    def _deduplicate_by_image_content(
        self,
        annotations: List[BBoxAnnotation],
        image_dir: Path
    ) -> List[BBoxAnnotation]:
        """基于图像内容去重"""
        # 计算每个图像的hash
        image_hashes = {}
        for ann in annotations:
            image_path = image_dir / ann.crop_path
            if image_path.exists():
                try:
                    img_hash = self._compute_image_hash(image_path)
                    image_hashes[ann] = img_hash
                except Exception as e:
                    logger.error(f"计算图像hash失败 {image_path}: {e}")
        
        # 基于hash去重
        unique_annotations = []
        seen_hashes = {}
        
        for ann in annotations:
            if ann not in image_hashes:
                # 无法计算hash的保留
                unique_annotations.append(ann)
                continue
            
            ann_hash = image_hashes[ann]
            is_duplicate = False
            
            # 检查是否与已见过的图像相似
            for seen_ann, seen_hash in seen_hashes.items():
                distance = ann_hash - seen_hash
                if distance <= self.max_distance:
                    # 检查caption相似度作为额外验证
                    if self._similar_captions(ann.caption, seen_ann.caption):
                        is_duplicate = True
                        logger.debug(
                            f"图像内容重复: {ann.crop_path} 与 {seen_ann.crop_path} "
                            f"距离={distance}"
                        )
                        break
            
            if not is_duplicate:
                seen_hashes[ann] = ann_hash
                unique_annotations.append(ann)
        
        logger.info(
            f"图像去重: {len(annotations)} -> {len(unique_annotations)} "
            f"(移除{len(annotations) - len(unique_annotations)}个重复)"
        )
        
        return unique_annotations
    
    def _compute_iou(self, bbox1, bbox2) -> float:
        """计算IoU"""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_image_hash(self, image_path: Path):
        """计算图像的感知hash"""
        img = Image.open(image_path)
        # 使用average hash，对小的变化更鲁棒
        return imagehash.average_hash(img, hash_size=self.hash_size)
    
    def _similar_captions(self, caption1: Optional[str], caption2: Optional[str]) -> bool:
        """检查两个caption是否相似"""
        if not caption1 or not caption2:
            return False
        
        # 简单的相似度检查
        words1 = set(caption1.lower().split())
        words2 = set(caption2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) > 0.8


class DatasetDeduplicator:
    """数据集去重器"""
    
    def __init__(self):
        self.text_dedup = TextDeduplicator()
        self.image_dedup = ImageDeduplicator()
    
    def deduplicate_dataset(
        self,
        documents: List[DocumentAnnotation],
        bbox_annotations: List[BBoxAnnotation],
        image_dir: Path
    ) -> Tuple[List[DocumentAnnotation], List[BBoxAnnotation]]:
        """去重整个数据集"""
        logger.info("开始数据集去重...")
        
        # 1. 去重文档
        unique_docs = self.text_dedup.deduplicate_documents(documents)
        valid_paper_ids = {doc.paper_id for doc in unique_docs}
        
        # 2. 过滤无效paper_id的bbox标注
        valid_bboxes = [
            bbox for bbox in bbox_annotations
            if bbox.paper_id in valid_paper_ids
        ]
        
        # 3. 去重bbox标注
        unique_bboxes = self.image_dedup.deduplicate_bbox_annotations(
            valid_bboxes,
            image_dir
        )
        
        logger.info(
            f"去重完成: 文档 {len(documents)} -> {len(unique_docs)}, "
            f"边界框 {len(bbox_annotations)} -> {len(unique_bboxes)}"
        )
        
        return unique_docs, unique_bboxes