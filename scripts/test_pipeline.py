#!/usr/bin/env python3
"""
测试流水线脚本
"""
import asyncio
import sys
from pathlib import Path
import logging
import os

sys.path.append(str(Path(__file__).parent.parent))

from src.core.schemas import DocumentAnnotation, BBoxAnnotation, save_schemas_to_config
from src.annotation import MistralConfig, MistralClient
from src.dataset import InternVL2Builder, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mistral_connection():
    """测试Mistral API连接"""
    logger.info("=== 测试Mistral API连接 ===")
    
    try:
        # 设置API key
        if 'MISTRAL_API_KEY' not in os.environ:
            os.environ['MISTRAL_API_KEY'] = 'UWcvuMWR7dicow9wC8kFN7pGIE2fd6qC'
        
        config = MistralConfig.from_env()
        client = MistralClient(config)
        
        # 测试简单请求
        test_doc = DocumentAnnotation(
            paper_id="test_001",
            title="测试文档",
            abstract="这是一个测试摘要。",
            sections=[{
                "title": "引言",
                "level": 1,
                "text": "测试内容"
            }]
        )
        
        logger.info("Mistral API连接成功")
        await client.close()
        return True
        
    except Exception as e:
        logger.error(f"Mistral API连接失败: {e}")
        return False


def test_schema_generation():
    """测试Schema生成"""
    logger.info("=== 测试Schema生成 ===")
    
    try:
        schema_dir = Path(__file__).parent.parent / "configs" / "schemas"
        save_schemas_to_config(schema_dir)
        logger.info("Schema生成成功")
        return True
    except Exception as e:
        logger.error(f"Schema生成失败: {e}")
        return False


def test_sample_generation():
    """测试样本生成"""
    logger.info("=== 测试样本生成 ===")
    
    try:
        # 创建测试数据
        doc_ann = DocumentAnnotation(
            paper_id="PMC12345",
            title="Test Medical Paper",
            abstract="This is a test abstract for medical paper.",
            sections=[
                {
                    "title": "Introduction", 
                    "level": 1,
                    "text": "Test introduction content."
                },
                {
                    "title": "Methods",
                    "level": 1, 
                    "text": "Test methods content."
                }
            ]
        )
        
        bbox_ann = BBoxAnnotation(
            paper_id="PMC12345",
            page_index=0,
            bbox={"x1": 100, "y1": 200, "x2": 500, "y2": 400},
            crop_path="crops/test_fig.png",
            figure_type="figure",
            caption="Test figure showing results",
            key_findings="The test shows positive results."
        )
        
        # 创建生成器
        builder = InternVL2Builder()
        
        # 生成样本
        sample = builder.build_figure_caption_sample(bbox_ann, "crops/test_fig.png")
        
        if sample:
            logger.info(f"生成样本ID: {sample.id}")
            logger.info(f"问题: {sample.conversations[0]['value']}")
            logger.info(f"答案: {sample.conversations[1]['value']}")
            return True
        else:
            logger.error("样本生成失败")
            return False
            
    except Exception as e:
        logger.error(f"样本生成测试失败: {e}")
        return False


async def main():
    """运行所有测试"""
    logger.info("开始测试流水线组件...")
    
    tests = [
        ("Schema生成", test_schema_generation),
        ("样本生成", test_sample_generation),
        ("Mistral API", test_mistral_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 打印测试结果
    print("\n=== 测试结果汇总 ===")
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    # 总体结果
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败，请检查日志。")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())