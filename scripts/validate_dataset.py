#!/usr/bin/env python3
"""
数据集验证工具
"""
import click
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import random
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.core.schemas import DocumentAnnotation, BBoxAnnotation
from src.dataset import InternVL2Sample
from src.quality import ConsistencyChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self):
        self.checker = ConsistencyChecker(strict_mode=True)
        self.stats = defaultdict(int)
        self.errors = []
        self.warnings = []
    
    def validate_jsonl(self, jsonl_path: Path, sample_ratio: float = 1.0) -> Dict[str, Any]:
        """验证JSONL数据集"""
        logger.info(f"验证数据集: {jsonl_path}")
        
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    self.errors.append(f"JSON解析错误: {e}")
        
        total_samples = len(samples)
        logger.info(f"总样本数: {total_samples}")
        
        # 采样验证
        if sample_ratio < 1.0:
            sample_size = int(total_samples * sample_ratio)
            samples = random.sample(samples, sample_size)
            logger.info(f"抽样验证 {sample_size} 个样本")
        
        # 验证每个样本
        valid_count = 0
        for i, sample in enumerate(samples):
            if self._validate_sample(sample, i):
                valid_count += 1
        
        # 统计信息
        self._compute_statistics(samples)
        
        # 生成报告
        report = {
            'total_samples': total_samples,
            'validated_samples': len(samples),
            'valid_samples': valid_count,
            'validation_rate': valid_count / len(samples) if samples else 0,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'statistics': dict(self.stats)
        }
        
        return report
    
    def _validate_sample(self, sample: Dict[str, Any], index: int) -> bool:
        """验证单个样本"""
        try:
            # 基本字段检查
            required_fields = ['id', 'image', 'conversations']
            for field in required_fields:
                if field not in sample:
                    self.errors.append(f"样本{index}缺少必需字段: {field}")
                    return False
            
            # 对话格式检查
            conversations = sample['conversations']
            if not self._validate_conversations(conversations):
                self.errors.append(f"样本{index}对话格式错误")
                return False
            
            # 图片数量检查
            image_count = 1 if isinstance(sample['image'], str) else len(sample['image'])
            full_text = " ".join([c['value'] for c in conversations])
            image_tag_count = full_text.count('<image>')
            
            if image_count != image_tag_count:
                self.errors.append(
                    f"样本{index}: 图片数量({image_count})与<image>标记({image_tag_count})不匹配"
                )
                return False
            
            # 宽高数据检查
            if image_count == 1:
                if 'width' not in sample or 'height' not in sample:
                    self.errors.append(f"样本{index}缺少width/height")
                    return False
            else:
                if 'width_list' not in sample or 'height_list' not in sample:
                    self.errors.append(f"样本{index}缺少width_list/height_list")
                    return False
                if len(sample['width_list']) != image_count or len(sample['height_list']) != image_count:
                    self.errors.append(f"样本{index}宽高列表长度不匹配")
                    return False
            
            # Grounding坐标检查
            if '<box>' in full_text:
                self._validate_grounding(full_text, sample, index)
            
            # 统计任务类型
            self._analyze_task_type(sample)
            
            return True
            
        except Exception as e:
            self.errors.append(f"样本{index}验证异常: {e}")
            return False
    
    def _validate_conversations(self, conversations: List[Dict]) -> bool:
        """验证对话格式"""
        if not conversations or len(conversations) % 2 != 0:
            return False
        
        for i, conv in enumerate(conversations):
            expected_from = 'human' if i % 2 == 0 else 'gpt'
            if conv.get('from') != expected_from:
                return False
        
        return True
    
    def _validate_grounding(self, text: str, sample: Dict, index: int):
        """验证grounding坐标"""
        import re
        
        # 提取所有box
        box_pattern = r'<box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        boxes = re.findall(box_pattern, text)
        
        # 获取图片尺寸
        if isinstance(sample['image'], str):
            widths = [sample.get('width', 0)]
            heights = [sample.get('height', 0)]
        else:
            widths = sample.get('width_list', [])
            heights = sample.get('height_list', [])
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取对应图片尺寸
            img_idx = min(i, len(widths) - 1)
            if img_idx < 0:
                self.warnings.append(f"样本{index}无法获取图片尺寸")
                continue
            
            width = widths[img_idx]
            height = heights[img_idx]
            
            # 检查坐标范围
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                self.errors.append(
                    f"样本{index}坐标越界: box[{x1},{y1},{x2},{y2}], "
                    f"图片尺寸[{width},{height}]"
                )
    
    def _analyze_task_type(self, sample: Dict):
        """分析任务类型"""
        conversations = sample['conversations']
        if not conversations:
            return
        
        question = conversations[0].get('value', '')
        answer = conversations[1].get('value', '') if len(conversations) > 1 else ''
        
        # 简单的任务类型判断
        if '<box>' in answer:
            self.stats['task_grounding'] += 1
        elif '表格' in question or 'table' in question.lower():
            self.stats['task_table'] += 1
        elif '变量' in question or 'variable' in question.lower():
            self.stats['task_variable'] += 1
        elif '比较' in question or '对比' in question:
            self.stats['task_multi_figure'] += 1
        else:
            self.stats['task_caption'] += 1
        
        # 统计其他信息
        if isinstance(sample['image'], list):
            self.stats['multi_image_samples'] += 1
        else:
            self.stats['single_image_samples'] += 1
    
    def _compute_statistics(self, samples: List[Dict]):
        """计算统计信息"""
        if not samples:
            return
        
        # 平均对话轮数
        conv_lengths = [len(s['conversations']) for s in samples]
        self.stats['avg_conversation_turns'] = sum(conv_lengths) / len(conv_lengths)
        
        # 平均问题长度
        question_lengths = []
        answer_lengths = []
        
        for s in samples:
            if s['conversations']:
                question_lengths.append(len(s['conversations'][0].get('value', '')))
                if len(s['conversations']) > 1:
                    answer_lengths.append(len(s['conversations'][1].get('value', '')))
        
        if question_lengths:
            self.stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        if answer_lengths:
            self.stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=== 数据集验证报告 ===\n")
        
        if self.errors:
            report.append(f"发现 {len(self.errors)} 个错误:")
            for i, error in enumerate(self.errors[:10]):  # 只显示前10个
                report.append(f"  {i+1}. {error}")
            if len(self.errors) > 10:
                report.append(f"  ... 还有 {len(self.errors) - 10} 个错误")
            report.append("")
        
        if self.warnings:
            report.append(f"发现 {len(self.warnings)} 个警告:")
            for i, warning in enumerate(self.warnings[:10]):
                report.append(f"  {i+1}. {warning}")
            if len(self.warnings) > 10:
                report.append(f"  ... 还有 {len(self.warnings) - 10} 个警告")
            report.append("")
        
        report.append("统计信息:")
        for key, value in sorted(self.stats.items()):
            if isinstance(value, float):
                report.append(f"  {key}: {value:.2f}")
            else:
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--sample-ratio', '-s', type=float, default=0.1,
              help='抽样验证比例 (0-1)')
@click.option('--output', '-o', type=click.Path(),
              help='输出报告路径')
@click.option('--full', is_flag=True, help='验证全部样本')
def main(dataset_path, sample_ratio, output, full):
    """验证InternVL2数据集"""
    if full:
        sample_ratio = 1.0
    
    validator = DatasetValidator()
    report = validator.validate_jsonl(Path(dataset_path), sample_ratio)
    
    # 打印摘要
    print(f"\n验证完成:")
    print(f"  总样本数: {report['total_samples']}")
    print(f"  验证样本数: {report['validated_samples']}")
    print(f"  有效样本数: {report['valid_samples']}")
    print(f"  验证通过率: {report['validation_rate']:.2%}")
    print(f"  错误数: {report['errors']}")
    print(f"  警告数: {report['warnings']}")
    
    # 打印详细报告
    detailed_report = validator.generate_report()
    print(f"\n{detailed_report}")
    
    # 保存报告
    if output:
        output_path = Path(output)
        
        # 保存JSON格式
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存文本格式
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        print(f"\n报告已保存到: {output_path}")


if __name__ == '__main__':
    main()