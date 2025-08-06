"""增强版QA模板 - 针对完整图表提取优化"""

from typing import List, Dict, Any
from enum import Enum
import random


class EnhancedTaskType(Enum):
    """增强的任务类型"""
    # 基础任务
    FIGURE_DETAILED_ANALYSIS = "figure_detailed_analysis"
    TABLE_COMPREHENSIVE_READING = "table_comprehensive_reading"
    CHART_DATA_EXTRACTION = "chart_data_extraction"
    
    # 高级任务
    FIGURE_COMPARISON = "figure_comparison"
    TREND_ANALYSIS = "trend_analysis"
    DATA_INTERPRETATION = "data_interpretation"
    VISUAL_REASONING = "visual_reasoning"
    
    # 医学特定任务
    MEDICAL_DIAGNOSIS_SUPPORT = "medical_diagnosis_support"
    CLINICAL_DATA_ANALYSIS = "clinical_data_analysis"


class EnhancedQATemplates:
    """增强版QA模板生成器"""
    
    def __init__(self):
        self.templates = self._init_templates()
    
    def _init_templates(self) -> Dict[EnhancedTaskType, List[Dict[str, Any]]]:
        """初始化所有模板"""
        return {
            EnhancedTaskType.FIGURE_DETAILED_ANALYSIS: [
                {
                    "question": "Provide a comprehensive analysis of this figure, including: 1) The type and purpose of the visualization, 2) All visible data elements and their relationships, 3) Key patterns or trends, 4) Statistical significance if shown, and 5) Clinical or research implications.",
                    "answer_format": "structured_analysis"
                },
                {
                    "question": "What are all the variables shown in this figure? For each variable, specify: its role (independent/dependent), measurement units, range of values, and any notable characteristics.",
                    "answer_format": "variable_listing"
                },
                {
                    "question": "Describe the visual elements of this figure in detail, including colors, shapes, legends, error bars, annotations, and any statistical indicators.",
                    "answer_format": "visual_description"
                }
            ],
            
            EnhancedTaskType.TABLE_COMPREHENSIVE_READING: [
                {
                    "question": "Extract all data from this table and provide: 1) Complete table structure in CSV format, 2) Summary of what each column represents, 3) Key findings or significant values, 4) Any footnotes or special notations.",
                    "answer_format": "table_full_extraction"
                },
                {
                    "question": "What are the main comparisons being made in this table? Identify the groups being compared, the metrics used, and highlight the most significant differences.",
                    "answer_format": "comparative_analysis"
                },
                {
                    "question": "Find all statistical measures in this table (p-values, confidence intervals, standard deviations, etc.) and explain their significance in the context of the study.",
                    "answer_format": "statistical_extraction"
                }
            ],
            
            EnhancedTaskType.CHART_DATA_EXTRACTION: [
                {
                    "question": "Extract the precise numerical values from this chart. For bar charts, provide heights; for line graphs, key points; for scatter plots, cluster centers or regression parameters.",
                    "answer_format": "numerical_extraction"
                },
                {
                    "question": "What is the scale and range of each axis in this chart? Include any logarithmic scales, break points, or non-linear transformations.",
                    "answer_format": "axis_analysis"
                }
            ],
            
            EnhancedTaskType.TREND_ANALYSIS: [
                {
                    "question": "Analyze the trends shown in this figure. Describe: 1) Overall direction of change, 2) Rate of change, 3) Any inflection points or plateaus, 4) Cyclical patterns if present, 5) Predictions based on the trend.",
                    "answer_format": "trend_comprehensive"
                },
                {
                    "question": "Identify any correlations visible in this figure. Specify the strength, direction, and potential causal relationships between variables.",
                    "answer_format": "correlation_analysis"
                }
            ],
            
            EnhancedTaskType.DATA_INTERPRETATION: [
                {
                    "question": "Based on the data shown in this figure, what conclusions can be drawn? Consider statistical significance, clinical relevance, and limitations of the data.",
                    "answer_format": "interpretation"
                },
                {
                    "question": "How does the data in this figure support or contradict the study's hypothesis? Provide specific evidence from the visualization.",
                    "answer_format": "hypothesis_evaluation"
                }
            ],
            
            EnhancedTaskType.MEDICAL_DIAGNOSIS_SUPPORT: [
                {
                    "question": "If this figure shows diagnostic imaging or clinical data, what pathological features or abnormalities can be identified? Describe their location, characteristics, and clinical significance.",
                    "answer_format": "diagnostic_analysis"
                },
                {
                    "question": "What clinical measurements or biomarkers are displayed in this figure? Specify normal ranges and interpret any deviations.",
                    "answer_format": "clinical_metrics"
                }
            ],
            
            EnhancedTaskType.VISUAL_REASONING: [
                {
                    "question": "Compare the different panels or subplots in this figure. What is the relationship between them, and what story do they tell together?",
                    "answer_format": "multi_panel_analysis"
                },
                {
                    "question": "If you had to explain this figure to a patient or non-specialist, what would be the key takeaway messages?",
                    "answer_format": "simplified_explanation"
                }
            ]
        }
    
    def get_questions_for_task(self, task_type: EnhancedTaskType, 
                             figure_type: str = None,
                             has_multiple_panels: bool = False,
                             is_medical_image: bool = False) -> List[str]:
        """根据任务类型和图表特征获取合适的问题"""
        questions = [t["question"] for t in self.templates.get(task_type, [])]
        
        # 根据图表特征调整问题
        if figure_type == "table" and task_type != EnhancedTaskType.TABLE_COMPREHENSIVE_READING:
            # 过滤掉不适合表格的问题
            questions = [q for q in questions if "axis" not in q.lower() and "chart" not in q.lower()]
        
        if has_multiple_panels:
            # 添加多面板相关问题
            questions.append("Describe the relationship between the different panels in this figure and how they complement each other.")
        
        if is_medical_image:
            # 添加医学影像相关问题
            questions.append("What imaging modality is shown and what anatomical structures or pathological features are visible?")
        
        return questions
    
    def build_enhanced_answer(self, task_type: EnhancedTaskType,
                            figure_data: Dict[str, Any],
                            ocr_results: Dict[str, Any] = None,
                            context: str = None) -> str:
        """构建增强的答案"""
        
        if task_type == EnhancedTaskType.FIGURE_DETAILED_ANALYSIS:
            return self._build_detailed_analysis_answer(figure_data, ocr_results, context)
        
        elif task_type == EnhancedTaskType.TABLE_COMPREHENSIVE_READING:
            return self._build_comprehensive_table_answer(figure_data, ocr_results)
        
        elif task_type == EnhancedTaskType.CHART_DATA_EXTRACTION:
            return self._build_data_extraction_answer(figure_data, ocr_results)
        
        elif task_type == EnhancedTaskType.TREND_ANALYSIS:
            return self._build_trend_analysis_answer(figure_data, ocr_results, context)
        
        # 其他任务类型...
        return self._build_generic_answer(figure_data, context)
    
    def _build_detailed_analysis_answer(self, figure_data: Dict[str, Any], 
                                      ocr_results: Dict[str, Any],
                                      context: str) -> str:
        """构建详细分析答案"""
        answer_parts = []
        
        # 1. 图表类型和目的
        fig_type = figure_data.get('figure_type', 'figure')
        answer_parts.append(f"This is a {fig_type} that appears to show {figure_data.get('caption', 'data visualization')}.")
        
        # 2. 数据元素和关系
        if 'variables' in figure_data:
            vars_desc = []
            for var in figure_data['variables']:
                role = var.get('role', 'variable')
                unit = var.get('unit', '')
                vars_desc.append(f"{var['name']} ({role}){' in ' + unit if unit else ''}")
            answer_parts.append(f"The figure displays the following variables: {', '.join(vars_desc)}.")
        
        # 3. 关键模式或趋势
        if 'key_findings' in figure_data:
            answer_parts.append(f"Key patterns observed: {figure_data['key_findings']}")
        
        # 4. 统计显著性
        if ocr_results and 'statistical_text' in ocr_results:
            answer_parts.append(f"Statistical indicators: {ocr_results['statistical_text']}")
        
        # 5. 临床或研究意义
        if context:
            answer_parts.append(f"In the context of this study, {context}")
        
        return " ".join(answer_parts)
    
    def _build_comprehensive_table_answer(self, figure_data: Dict[str, Any],
                                        ocr_results: Dict[str, Any]) -> str:
        """构建全面的表格答案"""
        answer_parts = []
        
        # CSV格式的表格数据
        if 'table_csv' in figure_data:
            answer_parts.append("Table data in CSV format:")
            answer_parts.append(figure_data['table_csv'])
        
        # 列描述
        if 'column_descriptions' in figure_data:
            answer_parts.append("\nColumn descriptions:")
            for col, desc in figure_data['column_descriptions'].items():
                answer_parts.append(f"- {col}: {desc}")
        
        # 关键发现
        if 'significant_values' in figure_data:
            answer_parts.append("\nSignificant findings:")
            answer_parts.append(figure_data['significant_values'])
        
        # 脚注
        if 'footnotes' in figure_data:
            answer_parts.append("\nFootnotes:")
            answer_parts.append(figure_data['footnotes'])
        
        return "\n".join(answer_parts)
    
    def _build_data_extraction_answer(self, figure_data: Dict[str, Any],
                                    ocr_results: Dict[str, Any]) -> str:
        """构建数据提取答案"""
        answer_parts = []
        
        # 提取的数值
        if 'extracted_values' in figure_data:
            answer_parts.append("Extracted numerical values:")
            for item in figure_data['extracted_values']:
                answer_parts.append(f"- {item['label']}: {item['value']} {item.get('unit', '')}")
        
        # 轴信息
        if 'axes' in figure_data:
            answer_parts.append("\nAxis information:")
            for axis_name, axis_info in figure_data['axes'].items():
                scale_type = axis_info.get('scale', 'linear')
                range_info = f"[{axis_info.get('min', 'N/A')} to {axis_info.get('max', 'N/A')}]"
                answer_parts.append(f"- {axis_name}: {axis_info.get('label', 'N/A')} {range_info}, {scale_type} scale")
        
        return "\n".join(answer_parts)
    
    def _build_trend_analysis_answer(self, figure_data: Dict[str, Any],
                                   ocr_results: Dict[str, Any],
                                   context: str) -> str:
        """构建趋势分析答案"""
        answer_parts = []
        
        # 整体趋势
        if 'overall_trend' in figure_data:
            answer_parts.append(f"Overall trend: {figure_data['overall_trend']}")
        
        # 变化率
        if 'rate_of_change' in figure_data:
            answer_parts.append(f"Rate of change: {figure_data['rate_of_change']}")
        
        # 拐点
        if 'inflection_points' in figure_data:
            answer_parts.append(f"Inflection points observed at: {figure_data['inflection_points']}")
        
        # 相关性
        if 'correlations' in figure_data:
            answer_parts.append(f"Correlations: {figure_data['correlations']}")
        
        return " ".join(answer_parts)
    
    def _build_generic_answer(self, figure_data: Dict[str, Any], context: str) -> str:
        """构建通用答案"""
        caption = figure_data.get('caption', 'No caption available')
        answer = f"Based on the figure showing '{caption}'"
        
        if context:
            answer += f", and considering the context: {context}"
        
        answer += ", the analysis would require more specific information about the visualization type and data shown."
        
        return answer


def create_enhanced_qa_pair(figure_data: Dict[str, Any],
                          task_type: EnhancedTaskType,
                          ocr_results: Dict[str, Any] = None,
                          context: str = None) -> Dict[str, str]:
    """创建增强的QA对"""
    generator = EnhancedQATemplates()
    
    # 确定图表特征
    figure_type = figure_data.get('figure_type', 'figure')
    has_multiple_panels = figure_data.get('has_multiple_panels', False)
    is_medical_image = figure_data.get('is_medical_image', False)
    
    # 获取问题
    questions = generator.get_questions_for_task(
        task_type, 
        figure_type,
        has_multiple_panels,
        is_medical_image
    )
    
    # 随机选择一个问题
    question = random.choice(questions) if questions else "Describe this figure in detail."
    
    # 生成答案
    answer = generator.build_enhanced_answer(task_type, figure_data, ocr_results, context)
    
    return {
        "question": question,
        "answer": answer,
        "task_type": task_type.value,
        "metadata": {
            "figure_type": figure_type,
            "has_ocr": ocr_results is not None,
            "has_context": context is not None
        }
    }