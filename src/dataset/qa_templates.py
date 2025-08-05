"""Q/A模板库"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import re

from ..core.schemas import (
    DocumentAnnotation,
    BBoxAnnotation,
    FigureType,
    VariableRole
)
from ..core.schemas.base import VariableRole as VarRole


class TaskType(str, Enum):
    """任务类型"""
    PAGE_GROUNDING = "page_grounding"  # 页面定位
    FIGURE_CAPTION = "figure_caption"  # 图表标题摘要
    VARIABLE_EXTRACTION = "variable_extraction"  # 变量/单位提取
    TABLE_READING = "table_reading"  # 表格读数
    MULTI_FIGURE = "multi_figure"  # 多图对比
    ABSTRACT_QA = "abstract_qa"  # 摘要问答


@dataclass
class QATemplate:
    """Q/A模板"""
    task_type: TaskType
    question_templates: List[str]
    answer_builder: callable
    required_fields: List[str]
    
    def generate_question(self, **kwargs) -> str:
        """生成问题"""
        template = random.choice(self.question_templates)
        return template.format(**kwargs)
    
    def generate_answer(self, data: Any) -> str:
        """生成答案"""
        return self.answer_builder(data)


class TemplateLibrary:
    """模板库"""
    
    def __init__(self):
        self.templates = self._init_templates()
    
    def _init_templates(self) -> Dict[TaskType, QATemplate]:
        """初始化模板"""
        return {
            TaskType.PAGE_GROUNDING: QATemplate(
                task_type=TaskType.PAGE_GROUNDING,
                question_templates=[
                    "请在页面中找到关于{topic}的章节标题，并标出其位置。",
                    "页面中{section_title}这一节在哪里？请用边界框标注。",
                    "请定位页面中的{element_type}，并返回其坐标。"
                ],
                answer_builder=self._build_grounding_answer,
                required_fields=['bbox', 'text']
            ),
            
            TaskType.FIGURE_CAPTION: QATemplate(
                task_type=TaskType.FIGURE_CAPTION,
                question_templates=[
                    "请用3-5句话总结这个{figure_type}的主要内容。",
                    "这个{figure_type}展示了什么？请简要描述其变量、趋势和主要结论。",
                    "请描述这个{figure_type}的内容，包括坐标轴含义和单位（如果有）。"
                ],
                answer_builder=self._build_figure_caption_answer,
                required_fields=['figure_type', 'caption', 'variables', 'axis', 'key_findings']
            ),
            
            TaskType.VARIABLE_EXTRACTION: QATemplate(
                task_type=TaskType.VARIABLE_EXTRACTION,
                question_templates=[
                    "这个图表中有哪些变量？它们的单位是什么？",
                    "请列出图中所有的变量名称、角色（自变量/因变量）和单位。",
                    "图表的横纵坐标分别代表什么？单位是什么？"
                ],
                answer_builder=self._build_variable_answer,
                required_fields=['variables', 'axis']
            ),
            
            TaskType.TABLE_READING: QATemplate(
                task_type=TaskType.TABLE_READING,
                question_templates=[
                    "表格中{row}行{column}列的值是多少？",
                    "请读取表格中{condition}条件下的数据。",
                    "将这个表格转换为CSV格式。"
                ],
                answer_builder=self._build_table_answer,
                required_fields=['table_csv', 'caption']
            ),
            
            TaskType.MULTI_FIGURE: QATemplate(
                task_type=TaskType.MULTI_FIGURE,
                question_templates=[
                    "比较这{num}个图表，它们的主要区别是什么？",
                    "这些图表展示了什么趋势变化？请对比分析。",
                    "综合这些图表，可以得出什么结论？"
                ],
                answer_builder=self._build_multi_figure_answer,
                required_fields=['figures']
            ),
            
            TaskType.ABSTRACT_QA: QATemplate(
                task_type=TaskType.ABSTRACT_QA,
                question_templates=[
                    "这篇论文的主要研究问题是什么？",
                    "论文的核心贡献和创新点是什么？",
                    "研究方法是什么？主要发现有哪些？"
                ],
                answer_builder=self._build_abstract_answer,
                required_fields=['abstract', 'sections']
            )
        }
    
    def _build_grounding_answer(self, data: Dict) -> str:
        """构建定位任务答案"""
        text = data.get('text', '')
        bbox = data.get('bbox')
        
        if bbox:
            return f"<ref>{text}</ref><box>{bbox}</box>"
        else:
            return f"未找到相关内容。"
    
    def _build_figure_caption_answer(self, data: BBoxAnnotation) -> str:
        """构建图表摘要答案"""
        parts = []
        
        # 图表类型
        type_map = {
            FigureType.FIGURE: "图表",
            FigureType.TABLE: "表格",
            FigureType.EQUATION: "公式",
            FigureType.DIAGRAM: "示意图",
            FigureType.FLOWCHART: "流程图"
        }
        parts.append(f"这是一个{type_map.get(data.figure_type, '图像')}。")
        
        # 变量描述
        if data.variables:
            var_desc = []
            for var in data.variables:
                if var.role in [VariableRole.X, VariableRole.Y]:
                    role_name = "横轴" if var.role == VariableRole.X else "纵轴"
                    unit_str = f"（单位：{var.unit}）" if var.unit else ""
                    var_desc.append(f"{role_name}为{var.name}{unit_str}")
            if var_desc:
                parts.append("，".join(var_desc) + "。")
        
        # 坐标轴信息
        if data.axis:
            axis_parts = []
            if data.axis.x_label:
                unit_str = f"（{data.axis.x_unit}）" if data.axis.x_unit else ""
                axis_parts.append(f"横轴：{data.axis.x_label}{unit_str}")
            if data.axis.y_label:
                unit_str = f"（{data.axis.y_unit}）" if data.axis.y_unit else ""
                axis_parts.append(f"纵轴：{data.axis.y_label}{unit_str}")
            if axis_parts:
                parts.append("；".join(axis_parts) + "。")
        
        # 关键发现
        if data.key_findings:
            parts.append(data.key_findings)
        
        # 如果信息不足，使用caption
        if len(parts) < 3 and data.caption:
            # 截断caption前80字
            caption_text = data.caption[:80]
            if len(data.caption) > 80:
                caption_text += "..."
            parts.append(caption_text)
        
        return " ".join(parts)
    
    def _build_variable_answer(self, data: BBoxAnnotation) -> str:
        """构建变量提取答案"""
        parts = []
        
        if data.variables:
            parts.append("图表中的变量包括：")
            for var in data.variables:
                role_map = {
                    VariableRole.X: "自变量",
                    VariableRole.Y: "因变量",
                    VariableRole.GROUP: "分组变量",
                    VariableRole.SERIES: "系列变量"
                }
                role_str = role_map.get(var.role, "变量")
                unit_str = f"，单位：{var.unit}" if var.unit else "，无单位"
                parts.append(f"- {var.name}（{role_str}{unit_str}）")
        
        if data.axis:
            if data.axis.x_label:
                unit_str = f"，单位：{data.axis.x_unit}" if data.axis.x_unit else ""
                parts.append(f"横坐标：{data.axis.x_label}{unit_str}")
            if data.axis.y_label:
                unit_str = f"，单位：{data.axis.y_unit}" if data.axis.y_unit else ""
                parts.append(f"纵坐标：{data.axis.y_label}{unit_str}")
        
        if not parts:
            return "该图表未标注变量信息。"
        
        return "\n".join(parts)
    
    def _build_table_answer(self, data: BBoxAnnotation) -> str:
        """构建表格答案"""
        if data.table_csv:
            return f"表格数据（CSV格式）：\n```csv\n{data.table_csv}\n```"
        else:
            return "无法提取表格的结构化数据。"
    
    def _build_multi_figure_answer(self, data: Dict) -> str:
        """构建多图对比答案"""
        figures = data.get('figures', [])
        if not figures:
            return "没有提供图表进行对比。"
        
        # 这里需要更复杂的逻辑来对比多个图表
        # 暂时返回简单描述
        parts = [f"共有{len(figures)}个图表进行对比。"]
        
        for i, fig in enumerate(figures):
            if hasattr(fig, 'key_findings') and fig.key_findings:
                parts.append(f"图{i+1}：{fig.key_findings}")
        
        return "\n".join(parts)
    
    def _build_abstract_answer(self, data: DocumentAnnotation) -> str:
        """构建摘要问答答案"""
        # 从摘要中提取关键信息
        abstract = data.abstract
        
        # 简单的关键句提取（实际应用中可以更复杂）
        sentences = re.split(r'[。！？.!?]', abstract)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        return "。".join(key_sentences) + "。"
    
    def get_template(self, task_type: TaskType) -> QATemplate:
        """获取模板"""
        return self.templates.get(task_type)
    
    def generate_qa_pair(
        self,
        task_type: TaskType,
        data: Any,
        **kwargs
    ) -> Tuple[str, str]:
        """生成Q/A对"""
        template = self.get_template(task_type)
        if not template:
            raise ValueError(f"未知的任务类型: {task_type}")
        
        # 检查必需字段
        for field in template.required_fields:
            if not hasattr(data, field) and field not in kwargs:
                raise ValueError(f"缺少必需字段: {field}")
        
        # 生成问题
        question = template.generate_question(**kwargs)
        
        # 生成答案
        answer = template.generate_answer(data)
        
        return question, answer


class QAGenerator:
    """Q/A生成器"""
    
    def __init__(self):
        self.template_library = TemplateLibrary()
    
    def generate_for_document(
        self,
        doc_annotation: DocumentAnnotation,
        task_types: List[TaskType] = None
    ) -> List[Dict[str, Any]]:
        """为文档生成Q/A对"""
        if task_types is None:
            task_types = [TaskType.ABSTRACT_QA]
        
        qa_pairs = []
        
        for task_type in task_types:
            try:
                question, answer = self.template_library.generate_qa_pair(
                    task_type,
                    doc_annotation
                )
                qa_pairs.append({
                    'task_type': task_type.value,
                    'question': question,
                    'answer': answer,
                    'metadata': {
                        'paper_id': doc_annotation.paper_id,
                        'source': 'document'
                    }
                })
            except Exception as e:
                print(f"生成{task_type}任务失败: {e}")
        
        return qa_pairs
    
    def generate_for_bbox(
        self,
        bbox_annotation: BBoxAnnotation,
        task_types: List[TaskType] = None
    ) -> List[Dict[str, Any]]:
        """为边界框生成Q/A对"""
        if task_types is None:
            # 根据图表类型选择任务
            if bbox_annotation.figure_type == FigureType.TABLE:
                task_types = [TaskType.FIGURE_CAPTION, TaskType.TABLE_READING]
            else:
                task_types = [TaskType.FIGURE_CAPTION, TaskType.VARIABLE_EXTRACTION]
        
        qa_pairs = []
        
        for task_type in task_types:
            try:
                # 准备参数
                kwargs = {
                    'figure_type': self._get_figure_type_name(bbox_annotation.figure_type)
                }
                
                question, answer = self.template_library.generate_qa_pair(
                    task_type,
                    bbox_annotation,
                    **kwargs
                )
                
                qa_pairs.append({
                    'task_type': task_type.value,
                    'question': question,
                    'answer': answer,
                    'metadata': {
                        'paper_id': bbox_annotation.paper_id,
                        'page_index': bbox_annotation.page_index,
                        'bbox': bbox_annotation.bbox.to_list(),
                        'figure_type': bbox_annotation.figure_type.value
                    }
                })
            except Exception as e:
                print(f"生成{task_type}任务失败: {e}")
        
        return qa_pairs
    
    def _get_figure_type_name(self, figure_type: FigureType) -> str:
        """获取图表类型的中文名"""
        name_map = {
            FigureType.FIGURE: "图表",
            FigureType.TABLE: "表格",
            FigureType.EQUATION: "公式",
            FigureType.DIAGRAM: "示意图",
            FigureType.FLOWCHART: "流程图",
            FigureType.OTHER: "图像"
        }
        return name_map.get(figure_type, "图像")