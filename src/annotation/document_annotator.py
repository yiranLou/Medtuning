"""文档级标注器"""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json

from .mistral_client import MistralClient
from .config import MistralConfig
from ..core.schemas import DocumentAnnotation, Section
from ..core.pdf_processor import PDFRenderer

logger = logging.getLogger(__name__)


class DocumentAnnotator:
    """文档级标注器，实现分批标注和防漂移策略"""
    
    def __init__(
        self, 
        mistral_client: Optional[MistralClient] = None,
        batch_pages: int = 5,
        overlap_pages: int = 1
    ):
        self.client = mistral_client or MistralClient()
        self.batch_pages = batch_pages  # 每批处理的页数
        self.overlap_pages = overlap_pages  # 批次间重叠页数
    
    async def annotate_document(
        self,
        pdf_path: Path,
        output_path: Optional[Path] = None,
        use_streaming: bool = True
    ) -> DocumentAnnotation:
        """标注整个文档"""
        pdf_path = Path(pdf_path)
        
        if use_streaming and self._get_page_count(pdf_path) > 10:
            # 大文档使用流式处理
            result = await self._annotate_streaming(pdf_path)
        else:
            # 小文档直接处理
            result = await self.client.annotate_document(pdf_path)
        
        # 保存结果
        if output_path:
            self._save_annotation(result, output_path)
        
        return result
    
    async def _annotate_streaming(self, pdf_path: Path) -> DocumentAnnotation:
        """流式标注大文档"""
        logger.info(f"开始流式标注文档: {pdf_path}")
        
        # 分批处理页面
        page_count = self._get_page_count(pdf_path)
        batches = self._create_page_batches(page_count)
        
        # 收集各批次结果
        all_sections = []
        paper_metadata = None
        
        for i, (start_page, end_page) in enumerate(batches):
            logger.info(f"处理批次 {i+1}/{len(batches)}: 页面 {start_page}-{end_page}")
            
            # 提取批次PDF
            batch_pdf = await self._extract_pages(pdf_path, start_page, end_page)
            
            # 构建批次提示
            batch_prompt = self._build_batch_prompt(i, len(batches), start_page, end_page)
            
            # 标注批次
            try:
                batch_result = await self.client.annotate_document(
                    batch_pdf,
                    additional_instructions=batch_prompt
                )
                
                # 合并结果
                if paper_metadata is None:
                    # 第一批提取元数据
                    paper_metadata = self._extract_metadata(batch_result)
                
                # 收集章节
                all_sections.extend(batch_result.sections)
                
            except Exception as e:
                logger.error(f"批次{i+1}处理失败: {e}")
                # 可以选择跳过或重试
                continue
        
        # 合并所有结果
        merged_result = self._merge_results(paper_metadata, all_sections)
        return merged_result
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """获取PDF页数"""
        with PDFRenderer(pdf_path) as renderer:
            return renderer.page_count
    
    def _create_page_batches(self, page_count: int) -> List[tuple]:
        """创建页面批次"""
        batches = []
        start = 0
        
        while start < page_count:
            end = min(start + self.batch_pages, page_count)
            batches.append((start, end))
            
            # 下一批起始位置（考虑重叠）
            start = end - self.overlap_pages
            if start >= page_count - self.overlap_pages:
                # 避免最后一批太小
                break
        
        return batches
    
    async def _extract_pages(self, pdf_path: Path, start: int, end: int) -> Path:
        """提取PDF的指定页面范围"""
        # TODO: 实现PDF页面提取
        # 这里需要使用PyMuPDF或其他库来提取特定页面
        # 暂时返回原PDF路径作为占位
        return pdf_path
    
    def _build_batch_prompt(
        self, 
        batch_idx: int, 
        total_batches: int,
        start_page: int,
        end_page: int
    ) -> str:
        """构建批次处理提示"""
        if batch_idx == 0:
            return f"""这是文档的第1批（共{total_batches}批），包含页面{start_page+1}-{end_page}。
请提取完整的元数据（标题、作者、摘要等）和这部分的章节内容。"""
        else:
            return f"""这是文档的第{batch_idx+1}批（共{total_batches}批），包含页面{start_page+1}-{end_page}。
只需要提取这部分的章节内容，不需要重复提取元数据。
注意保持章节编号的连续性。"""
    
    def _extract_metadata(self, annotation: DocumentAnnotation) -> Dict[str, Any]:
        """提取文档元数据"""
        return {
            'paper_id': annotation.paper_id,
            'title': annotation.title,
            'abstract': annotation.abstract,
            'keywords': annotation.keywords,
            'authors': annotation.authors,
            'affiliations': annotation.affiliations,
            'doi': annotation.doi,
            'journal': annotation.journal,
            'publication_date': annotation.publication_date
        }
    
    def _merge_results(
        self, 
        metadata: Dict[str, Any], 
        all_sections: List[Section]
    ) -> DocumentAnnotation:
        """合并批次结果"""
        # 去重和排序章节
        unique_sections = self._deduplicate_sections(all_sections)
        
        # 创建合并后的标注
        merged_data = {
            **metadata,
            'sections': unique_sections,
            'references': []  # 参考文献通常在最后，可能需要特殊处理
        }
        
        return DocumentAnnotation.model_validate(merged_data)
    
    def _deduplicate_sections(self, sections: List[Section]) -> List[Section]:
        """去重章节（处理重叠部分）"""
        # 基于标题和级别去重
        seen = set()
        unique = []
        
        for section in sections:
            key = (section.title, section.level)
            if key not in seen:
                seen.add(key)
                unique.append(section)
            else:
                # 如果是重复的，选择内容更长的版本
                for i, existing in enumerate(unique):
                    if (existing.title, existing.level) == key:
                        if len(section.text) > len(existing.text):
                            unique[i] = section
                        break
        
        return unique
    
    def _save_annotation(self, annotation: DocumentAnnotation, output_path: Path):
        """保存标注结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                annotation.model_dump(exclude_none=True),
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info(f"已保存文档标注到: {output_path}")


class DocumentAnnotationPostProcessor:
    """文档标注后处理器"""
    
    def process(self, annotation: DocumentAnnotation) -> DocumentAnnotation:
        """后处理文档标注"""
        # 1. 修复章节层级
        annotation = self._fix_section_hierarchy(annotation)
        
        # 2. 合并相邻段落
        annotation = self._merge_adjacent_paragraphs(annotation)
        
        # 3. 清理文本
        annotation = self._clean_texts(annotation)
        
        # 4. 提取缺失的参考文献
        annotation = self._extract_references(annotation)
        
        return annotation
    
    def _fix_section_hierarchy(self, annotation: DocumentAnnotation) -> DocumentAnnotation:
        """修复章节层级关系"""
        # TODO: 实现章节层级修复逻辑
        return annotation
    
    def _merge_adjacent_paragraphs(self, annotation: DocumentAnnotation) -> DocumentAnnotation:
        """合并相邻的同级段落"""
        # TODO: 实现段落合并逻辑
        return annotation
    
    def _clean_texts(self, annotation: DocumentAnnotation) -> DocumentAnnotation:
        """清理文本（去除页眉页脚等）"""
        # TODO: 实现文本清理逻辑
        return annotation
    
    def _extract_references(self, annotation: DocumentAnnotation) -> DocumentAnnotation:
        """提取参考文献（如果缺失）"""
        # TODO: 实现参考文献提取逻辑
        return annotation