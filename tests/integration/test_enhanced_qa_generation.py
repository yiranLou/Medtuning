#!/usr/bin/env python3
"""测试增强的QA生成系统"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from src.core.pdf_processor.working_enhanced_detector import WorkingEnhancedDetector
from src.core.pdf_processor.renderer import PDFRenderer
from src.core.dataset_builder.enhanced_qa_templates import (
    EnhancedTaskType, 
    create_enhanced_qa_pair
)


def demonstrate_qa_improvements():
    """演示QA生成的改进"""
    
    print("=== 增强QA生成系统演示 ===\n")
    
    # 示例1: 完整图表分析
    print("1. 图表详细分析示例:")
    figure_data_1 = {
        'figure_type': 'chart',
        'caption': 'Comparison of treatment outcomes across different patient groups',
        'variables': [
            {'name': 'Treatment Group', 'role': 'X', 'unit': ''},
            {'name': 'Response Rate', 'role': 'Y', 'unit': '%'},
            {'name': 'Patient Age', 'role': 'GROUP', 'unit': 'years'}
        ],
        'key_findings': 'Treatment A showed 85% response rate in younger patients vs 65% in older patients',
        'axes': {
            'x-axis': {'label': 'Treatment Groups', 'min': 0, 'max': 4},
            'y-axis': {'label': 'Response Rate (%)', 'min': 0, 'max': 100}
        }
    }
    
    qa_pair_1 = create_enhanced_qa_pair(
        figure_data_1,
        EnhancedTaskType.FIGURE_DETAILED_ANALYSIS,
        context="This study evaluates the efficacy of novel cancer treatments"
    )
    
    print(f"问题: {qa_pair_1['question']}")
    print(f"答案: {qa_pair_1['answer']}\n")
    
    # 示例2: 表格数据提取
    print("2. 表格全面读取示例:")
    figure_data_2 = {
        'figure_type': 'table',
        'caption': 'Patient demographics and baseline characteristics',
        'table_csv': '''Group,N,Age (mean±SD),Male (%),BMI (mean±SD)
Control,150,45.2±12.3,48.0,25.3±3.2
Treatment A,148,44.8±11.9,52.0,25.1±3.5
Treatment B,152,46.1±13.1,49.3,25.8±3.3''',
        'column_descriptions': {
            'Group': 'Study arm assignment',
            'N': 'Number of participants',
            'Age': 'Age in years at enrollment',
            'Male (%)': 'Percentage of male participants',
            'BMI': 'Body Mass Index'
        },
        'significant_values': 'No significant differences in baseline characteristics (p>0.05 for all comparisons)'
    }
    
    qa_pair_2 = create_enhanced_qa_pair(
        figure_data_2,
        EnhancedTaskType.TABLE_COMPREHENSIVE_READING
    )
    
    print(f"问题: {qa_pair_2['question']}")
    print(f"答案: {qa_pair_2['answer']}\n")
    
    # 示例3: 趋势分析
    print("3. 趋势分析示例:")
    figure_data_3 = {
        'figure_type': 'line_graph',
        'caption': 'Disease progression over 24-month follow-up',
        'overall_trend': 'Decreasing disease activity with treatment',
        'rate_of_change': '-2.5 units per month in treatment group vs -0.8 units per month in control',
        'inflection_points': 'Month 6 (treatment effect plateau), Month 18 (secondary improvement)',
        'correlations': 'Strong negative correlation between treatment duration and disease activity (r=-0.82, p<0.001)'
    }
    
    qa_pair_3 = create_enhanced_qa_pair(
        figure_data_3,
        EnhancedTaskType.TREND_ANALYSIS
    )
    
    print(f"问题: {qa_pair_3['question']}")
    print(f"答案: {qa_pair_3['answer']}\n")
    
    # 示例4: 数据解释
    print("4. 数据解释示例:")
    figure_data_4 = {
        'figure_type': 'box_plot',
        'caption': 'Distribution of treatment effects across subgroups',
        'key_findings': 'Median improvement: 45% (IQR: 35-55%), with significant outliers in genetic subgroup',
        'statistical_significance': 'p<0.001 for overall treatment effect, interaction p=0.023'
    }
    
    qa_pair_4 = create_enhanced_qa_pair(
        figure_data_4,
        EnhancedTaskType.DATA_INTERPRETATION,
        context="Primary endpoint was 40% improvement from baseline"
    )
    
    print(f"问题: {qa_pair_4['question']}")
    print(f"答案: {qa_pair_4['answer']}\n")


async def test_with_real_pdfs():
    """使用真实PDF测试增强的QA生成"""
    
    print("\n=== 使用真实PDF测试 ===\n")
    
    # PDF路径
    pdf_paths = [
        Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/PMC1301025.pdf"),
        Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/2309.09431v4.pdf")
    ]
    
    detector = WorkingEnhancedDetector()
    
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"PDF不存在: {pdf_path}")
            continue
        
        print(f"处理PDF: {pdf_path.name}")
        
        # 检测图表
        results = detector.detect_all_elements(pdf_path)
        
        # 为前3个图表生成增强QA对
        figures_to_process = results['figures'][:3]
        
        for i, figure in enumerate(figures_to_process):
            print(f"\n图表 {i+1} - {figure.caption}")
            
            # 构建图表数据
            figure_data = {
                'figure_type': 'figure',
                'caption': figure.caption,
                'page_number': figure.page_index + 1,
                'bbox': {
                    'x1': figure.bbox.x1,
                    'y1': figure.bbox.y1,
                    'x2': figure.bbox.x2,
                    'y2': figure.bbox.y2
                }
            }
            
            # 生成不同类型的QA对
            task_types = [
                EnhancedTaskType.FIGURE_DETAILED_ANALYSIS,
                EnhancedTaskType.VISUAL_REASONING
            ]
            
            for task_type in task_types:
                qa_pair = create_enhanced_qa_pair(
                    figure_data,
                    task_type,
                    context=f"From {pdf_path.stem}"
                )
                
                print(f"\n任务类型: {task_type.value}")
                print(f"问题: {qa_pair['question'][:100]}...")
                print(f"答案预览: {qa_pair['answer'][:150]}...")


def compare_old_vs_new():
    """比较旧版和新版QA生成"""
    
    print("\n=== 新旧版本对比 ===\n")
    
    figure_data = {
        'figure_type': 'chart',
        'caption': 'Clinical trial results',
        'variables': [
            {'name': 'Time', 'role': 'X', 'unit': 'months'},
            {'name': 'Survival Rate', 'role': 'Y', 'unit': '%'}
        ]
    }
    
    print("旧版QA（图片不完整时）:")
    print("问题: Summarize the figure.")
    print("答案: This appears to be a chart showing clinical trial results. Unable to extract detailed information due to incomplete image.\n")
    
    print("新版QA（图片完整时）:")
    qa_pair = create_enhanced_qa_pair(
        figure_data,
        EnhancedTaskType.FIGURE_DETAILED_ANALYSIS
    )
    print(f"问题: {qa_pair['question']}")
    print(f"答案: {qa_pair['answer']}\n")
    
    print("主要改进:")
    print("1. ✅ 完整图表 -> 准确的数据提取")
    print("2. ✅ 更多视觉信息 -> 更丰富的描述")
    print("3. ✅ 清晰的轴标签 -> 准确的变量识别")
    print("4. ✅ 完整的图例 -> 正确的数据解释")
    print("5. ✅ OCR支持 -> 表格数据的精确提取")


if __name__ == "__main__":
    # 运行演示
    demonstrate_qa_improvements()
    
    # 比较新旧版本
    compare_old_vs_new()
    
    # 测试真实PDF
    asyncio.run(test_with_real_pdfs())