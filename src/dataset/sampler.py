"""数据采样策略"""
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from pathlib import Path
import json
import logging

from .qa_templates import TaskType

logger = logging.getLogger(__name__)


class DatasetSampler:
    """数据集采样器"""
    
    def __init__(
        self,
        task_weights: Optional[Dict[TaskType, float]] = None,
        max_samples_per_paper: int = 50,
        min_samples_per_task: int = 100,
        balance_papers: bool = True
    ):
        self.task_weights = task_weights or self._default_task_weights()
        self.max_samples_per_paper = max_samples_per_paper
        self.min_samples_per_task = min_samples_per_task
        self.balance_papers = balance_papers
    
    def _default_task_weights(self) -> Dict[TaskType, float]:
        """默认任务权重"""
        return {
            TaskType.PAGE_GROUNDING: 0.15,      # 页面定位
            TaskType.FIGURE_CAPTION: 0.40,      # 图表摘要（重点）
            TaskType.VARIABLE_EXTRACTION: 0.15,  # 变量提取
            TaskType.TABLE_READING: 0.15,       # 表格读取
            TaskType.MULTI_FIGURE: 0.10,        # 多图对比
            TaskType.ABSTRACT_QA: 0.05          # 摘要问答
        }
    
    def sample_dataset(
        self,
        all_samples: List[Dict[str, Any]],
        target_size: int,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """采样数据集"""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 按任务类型和论文分组
        task_groups = defaultdict(list)
        paper_groups = defaultdict(list)
        
        for sample in all_samples:
            task_type = sample.get('task_type', 'unknown')
            paper_id = sample.get('metadata', {}).get('paper_id', 'unknown')
            
            task_groups[task_type].append(sample)
            paper_groups[paper_id].append(sample)
        
        # 1. 首先确保每个任务的最小样本数
        selected_samples = []
        selected_ids = set()
        
        for task_type, weight in self.task_weights.items():
            task_samples = task_groups.get(task_type.value, [])
            if not task_samples:
                continue
            
            # 计算该任务的目标样本数
            target_task_size = max(
                self.min_samples_per_task,
                int(target_size * weight)
            )
            
            # 采样
            sampled = self._sample_from_task(
                task_samples,
                target_task_size,
                selected_ids
            )
            selected_samples.extend(sampled)
            selected_ids.update(s.get('id', '') for s in sampled)
        
        # 2. 如果还需要更多样本，按权重补充
        remaining = target_size - len(selected_samples)
        if remaining > 0:
            # 创建候选池（排除已选择的）
            candidate_pool = [
                s for s in all_samples 
                if s.get('id', '') not in selected_ids
            ]
            
            # 按任务权重采样
            additional = self._weighted_sample(
                candidate_pool,
                remaining,
                self.task_weights
            )
            selected_samples.extend(additional)
        
        # 3. 如果需要平衡论文，进行调整
        if self.balance_papers:
            selected_samples = self._balance_by_paper(
                selected_samples,
                self.max_samples_per_paper
            )
        
        # 打乱顺序
        random.shuffle(selected_samples)
        
        # 记录统计信息
        self._log_statistics(selected_samples)
        
        return selected_samples[:target_size]
    
    def _sample_from_task(
        self,
        task_samples: List[Dict],
        target_size: int,
        excluded_ids: set
    ) -> List[Dict]:
        """从特定任务中采样"""
        # 过滤已选择的
        available = [
            s for s in task_samples 
            if s.get('id', '') not in excluded_ids
        ]
        
        if len(available) <= target_size:
            return available
        
        # 优先选择高质量样本（如果有置信度分数）
        if all('confidence_score' in s.get('metadata', {}) for s in available):
            # 按置信度排序
            available.sort(
                key=lambda s: s['metadata'].get('confidence_score', 0),
                reverse=True
            )
            # 选择前80%的高质量样本
            high_quality_count = int(target_size * 0.8)
            selected = available[:high_quality_count]
            
            # 从剩余的随机选择
            remaining = available[high_quality_count:]
            if remaining and target_size - high_quality_count > 0:
                additional = random.sample(
                    remaining,
                    min(len(remaining), target_size - high_quality_count)
                )
                selected.extend(additional)
            
            return selected
        else:
            # 随机采样
            return random.sample(available, target_size)
    
    def _weighted_sample(
        self,
        samples: List[Dict],
        target_size: int,
        weights: Dict[TaskType, float]
    ) -> List[Dict]:
        """按权重采样"""
        if not samples:
            return []
        
        # 计算每个样本的权重
        sample_weights = []
        for sample in samples:
            task_type = sample.get('task_type', 'unknown')
            weight = weights.get(TaskType(task_type), 0.1) if task_type != 'unknown' else 0.1
            sample_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(sample_weights)
        if total_weight > 0:
            sample_weights = [w / total_weight for w in sample_weights]
        else:
            sample_weights = [1.0 / len(samples)] * len(samples)
        
        # 按权重采样
        selected_indices = np.random.choice(
            len(samples),
            size=min(target_size, len(samples)),
            replace=False,
            p=sample_weights
        )
        
        return [samples[i] for i in selected_indices]
    
    def _balance_by_paper(
        self,
        samples: List[Dict],
        max_per_paper: int
    ) -> List[Dict]:
        """平衡每篇论文的样本数"""
        # 按论文分组
        paper_groups = defaultdict(list)
        for sample in samples:
            paper_id = sample.get('metadata', {}).get('paper_id', 'unknown')
            paper_groups[paper_id].append(sample)
        
        # 限制每篇论文的样本数
        balanced_samples = []
        for paper_id, paper_samples in paper_groups.items():
            if len(paper_samples) <= max_per_paper:
                balanced_samples.extend(paper_samples)
            else:
                # 保持任务多样性
                task_groups = defaultdict(list)
                for s in paper_samples:
                    task_type = s.get('task_type', 'unknown')
                    task_groups[task_type].append(s)
                
                # 从每个任务中选择
                selected = []
                samples_per_task = max_per_paper // len(task_groups)
                
                for task_samples in task_groups.values():
                    n = min(len(task_samples), samples_per_task)
                    selected.extend(random.sample(task_samples, n))
                
                # 如果还有空间，随机补充
                remaining = max_per_paper - len(selected)
                if remaining > 0:
                    unselected = [
                        s for s in paper_samples 
                        if s not in selected
                    ]
                    if unselected:
                        additional = random.sample(
                            unselected,
                            min(remaining, len(unselected))
                        )
                        selected.extend(additional)
                
                balanced_samples.extend(selected)
        
        return balanced_samples
    
    def _log_statistics(self, samples: List[Dict]):
        """记录统计信息"""
        # 任务分布
        task_counts = defaultdict(int)
        paper_counts = defaultdict(int)
        
        for sample in samples:
            task_type = sample.get('task_type', 'unknown')
            paper_id = sample.get('metadata', {}).get('paper_id', 'unknown')
            
            task_counts[task_type] += 1
            paper_counts[paper_id] += 1
        
        logger.info("=== 数据集统计 ===")
        logger.info(f"总样本数: {len(samples)}")
        
        logger.info("任务分布:")
        for task_type, count in sorted(task_counts.items()):
            percentage = count / len(samples) * 100
            logger.info(f"  {task_type}: {count} ({percentage:.1f}%)")
        
        logger.info(f"论文数: {len(paper_counts)}")
        logger.info(f"每篇论文平均样本数: {len(samples) / len(paper_counts):.1f}")
        logger.info(f"每篇论文最大样本数: {max(paper_counts.values())}")
        logger.info(f"每篇论文最小样本数: {min(paper_counts.values())}")


class MetaConfig:
    """元配置管理"""
    
    def __init__(self):
        self.config = {
            "dataset_info": {
                "version": "1.0",
                "description": "医学文献多模态理解数据集",
                "task_types": [t.value for t in TaskType],
                "languages": ["zh", "en"]
            },
            "training_config": {
                "max_dynamic_patch": 24,  # InternVL2动态切片上限
                "repeat_time": 1,         # 数据复用倍数
                "shuffle": True,
                "seed": 42
            },
            "data_sources": {}
        }
    
    def add_data_source(
        self,
        name: str,
        path: str,
        repeat_time: int = 1,
        max_dynamic_patch: int = 24,
        metadata: Optional[Dict] = None
    ):
        """添加数据源"""
        self.config["data_sources"][name] = {
            "path": path,
            "repeat_time": repeat_time,
            "max_dynamic_patch": max_dynamic_patch,
            "metadata": metadata or {}
        }
    
    def save(self, output_path: Path):
        """保存元配置"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存meta.json到: {output_path}")
    
    @classmethod
    def load(cls, config_path: Path) -> 'MetaConfig':
        """加载元配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        meta = cls()
        meta.config = config_data
        return meta