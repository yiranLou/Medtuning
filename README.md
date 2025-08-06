# MedTuning - 医学文献多模态数据集构建工具

## 目录
- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [代码架构详解](#代码架构详解)
- [使用示例](#使用示例)
- [输出说明](#输出说明)
- [配置详解](#配置详解)
- [常见问题](#常见问题)

## 项目概述

MedTuning 是一个专门为医学文献设计的多模态数据集构建工具，通过智能识别和提取PDF中的图表、表格等元素，结合 Mistral Document AI 进行高质量标注，生成适用于 InternVL2 模型微调的标准化数据集。

### 为什么选择 MedTuning？

1. **专注医学领域**：针对医学文献的特殊格式和内容进行优化
2. **高精度检测**：采用增强的检测算法，准确识别图表、表格和公式
3. **智能标注**：利用 Mistral AI 的强大能力，生成高质量的描述和问答对
4. **标准化输出**：完全兼容 InternVL2 的训练格式要求

## 核心功能

### 1. PDF 文档处理
- **高质量渲染**：支持 200-300 DPI 的高清图片输出
- **智能检测**：自动识别PDF中的图表、表格、公式等元素
- **精确裁剪**：根据检测结果精确提取各类元素

### 2. 元素检测技术
- **嵌入图片检测**：识别PDF中的所有嵌入式图片
- **表格识别**：基于文本布局和线条检测表格结构
- **坐标映射**：精确的PDF坐标到像素坐标转换

### 3. 智能标注系统
- **文档级标注**：生成整篇文档的摘要、关键词和主题
- **元素级标注**：为每个图表/表格生成详细描述
- **防漂移机制**：确保标注内容的准确性和一致性

### 4. 数据集生成
- **多任务支持**：
  - 页面定位（定位特定内容在哪一页）
  - 图表摘要（描述图表内容）
  - 变量提取（从图表中提取关键信息）
  - 表格读取（理解表格数据）
  - 多图对比（比较多个图表）
  - 摘要问答（基于文档内容的问答）
- **质量控制**：自动验证、去重和一致性检查

## 安装指南

### 系统要求
- Python 3.8+
- 足够的磁盘空间（处理大量PDF时需要存储图片）
- 推荐：GPU支持（加速图像处理）

### 安装步骤

1. 克隆项目
```bash
git clone <repository_url>
cd medtuning-master/new
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 设置环境变量
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

## 快速开始

### 1. 准备PDF文件
将医学文献PDF文件放置在 `data/raw_pdfs/` 目录下。

### 2. 运行测试脚本
```bash
# 测试单个PDF的处理
python test_pdf_processing.py /path/to/your.pdf

# 输出将保存在 output/test_extraction/ 目录
```

### 3. 运行完整流水线
```bash
# 使用默认配置
python scripts/run_enhanced_pipeline.py

# 指定配置文件
python scripts/run_enhanced_pipeline.py --config configs/config.yaml

# 处理特定PDF
python scripts/run_enhanced_pipeline.py --pdf-file /path/to/paper.pdf
```

## 代码架构详解

### 项目结构
```
new/
├── src/                          # 源代码目录
│   ├── core/                     # 核心功能模块
│   │   ├── pdf_processor/        # PDF处理相关
│   │   │   ├── __init__.py
│   │   │   ├── renderer.py       # PDF渲染器，负责页面和区域渲染
│   │   │   ├── working_enhanced_detector.py  # 增强检测器（当前主要版本）
│   │   │   └── enhanced_detector.py          # 另一个版本的检测器
│   │   ├── schemas/              # 数据模式定义
│   │   │   ├── base.py          # 基础模式类
│   │   │   ├── document.py      # 文档级模式
│   │   │   ├── bbox.py          # 边界框模式
│   │   │   └── json_schemas.py  # JSON Schema定义
│   │   └── validators/           # 数据验证器
│   │
│   ├── annotation/               # 标注模块
│   │   ├── mistral_client.py    # Mistral API客户端
│   │   ├── document_annotator.py # 文档级标注器
│   │   └── bbox_annotator.py    # 边界框标注器
│   │
│   ├── dataset/                  # 数据集构建模块
│   │   ├── internvl2_builder.py # InternVL2格式构建器
│   │   ├── qa_templates.py      # 问答模板
│   │   └── sampler.py           # 数据采样器
│   │
│   └── quality/                  # 质量控制模块
│       ├── consistency_checker.py # 一致性检查
│       └── deduplication.py      # 去重处理
│
├── scripts/                      # 可执行脚本
│   ├── run_enhanced_pipeline.py  # 主流水线脚本
│   ├── test_pipeline.py         # 测试脚本
│   └── validate_dataset.py      # 数据集验证
│
├── configs/                      # 配置文件
│   ├── config.yaml              # 主配置文件
│   ├── meta.json                # 元数据配置
│   └── templates/               # 模板文件
│
├── data/                        # 数据目录
│   ├── raw_pdfs/               # 原始PDF文件
│   ├── processed/              # 处理后的数据
│   └── outputs/                # 最终输出
│
└── output/                      # 临时输出目录
    └── test_extraction/         # 测试提取结果
```

### 核心模块详解

#### 1. PDF处理器 (pdf_processor)

**PDFRenderer 类** (`renderer.py`)
```python
class PDFRenderer:
    """PDF渲染器，负责将PDF转换为图片"""
    
    def __init__(self, pdf_path, config=None):
        # 初始化PyMuPDF文档对象
        # 设置渲染配置（DPI、颜色模式等）
    
    def render_page(self, page_index, output_path):
        # 渲染指定页面为图片
        # 支持自定义DPI和格式
    
    def crop_region(self, page_index, bbox, output_path):
        # 裁剪页面的指定区域
        # 自动处理边界扩展和缩放
```

**WorkingEnhancedDetector 类** (`working_enhanced_detector.py`)
```python
class WorkingEnhancedDetector:
    """增强的图表检测器"""
    
    def detect_all_elements(self, pdf_path):
        # 检测PDF中的所有元素
        # 返回：{'figures': [...], 'tables': [...], 'equations': [...]}
    
    def _detect_embedded_images(self, page, page_num):
        # 检测嵌入的图片
        # 使用PyMuPDF的get_images()方法
    
    def _detect_text_tables(self, page, page_num):
        # 基于文本布局检测表格
        # 分析文本块的对齐和间距
```

#### 2. 标注系统 (annotation)

**MistralClient 类** (`mistral_client.py`)
```python
class MistralClient:
    """Mistral API客户端，处理与AI的交互"""
    
    async def call_api(self, messages, schema=None):
        # 调用Mistral API
        # 支持重试和错误处理
        # 可选的JSON Schema验证
```

**DocumentAnnotator 类** (`document_annotator.py`)
```python
class DocumentAnnotator:
    """文档级标注器"""
    
    async def annotate_document(self, pdf_path):
        # 生成文档级标注
        # 包括：标题、摘要、关键词、主题分类等
```

#### 3. 数据集构建 (dataset)

**InternVL2Builder 类** (`internvl2_builder.py`)
```python
class InternVL2Builder:
    """构建InternVL2格式的数据集"""
    
    def build_from_annotations(self, doc_annotations, bbox_annotations):
        # 从标注构建数据集样本
        # 生成多种任务类型的问答对
    
    def save_to_jsonl(self, samples, output_path):
        # 保存为JSONL格式
        # 每行一个JSON对象
```

### 数据流程

1. **PDF输入** → 2. **元素检测** → 3. **图片提取** → 4. **AI标注** → 5. **数据集生成**

## 使用示例

### 示例1：处理单个PDF并提取图表

```python
from src.core.pdf_processor.working_enhanced_detector import WorkingEnhancedDetector
from src.core.pdf_processor import PDFRenderer, RenderConfig

# 初始化检测器
detector = WorkingEnhancedDetector()

# 检测PDF中的元素
results = detector.detect_all_elements("path/to/medical_paper.pdf")
print(f"检测到 {len(results['figures'])} 个图片")
print(f"检测到 {len(results['tables'])} 个表格")

# 提取检测到的图片
config = RenderConfig()
config.dpi = 300  # 高质量输出

with PDFRenderer("path/to/medical_paper.pdf", config) as renderer:
    for i, figure in enumerate(results['figures']):
        output_path = f"output/figure_{i}.png"
        renderer.crop_region(figure.page_index, figure.bbox, output_path)
```

### 示例2：生成InternVL2数据集

```python
# 运行完整流水线
python scripts/run_enhanced_pipeline.py \
    --pdf-dir data/raw_pdfs \
    --config configs/config.yaml

# 输出文件：
# - data/outputs/internvl2_dataset.jsonl  # 数据集文件
# - data/outputs/meta.json                # 元数据
# - data/processed/*/                     # 处理后的图片
```

## 输出说明

### 1. 图片输出位置

处理后的图片保存在以下目录结构中：

```
data/processed/
├── [paper_id]/                  # 每篇论文一个目录
│   ├── pages/                   # 完整页面图片
│   │   ├── page_0.png
│   │   ├── page_1.png
│   │   └── ...
│   ├── figures/                 # 提取的图表
│   │   ├── figure_0.png
│   │   ├── figure_1.png
│   │   └── ...
│   ├── tables/                  # 提取的表格
│   │   ├── table_0.png
│   │   └── ...
│   └── document_annotation.json # 文档标注结果
```

### 2. 数据集格式

生成的 JSONL 文件格式示例：

```json
{
  "id": "med_sample_001",
  "image": ["processed/PMC123/pages/page_0.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is shown in this medical figure?"
    },
    {
      "from": "gpt", 
      "value": "This figure shows a flow cytometry analysis..."
    }
  ]
}
```

### 3. 临时测试输出

运行测试脚本时，输出保存在：

```
output/test_extraction/
├── pages/       # PDF页面图片
├── figures/     # 提取的图表
└── tables/      # 提取的表格
```

## 配置详解

### config.yaml 主要配置项

```yaml
# PDF处理配置
pdf_processing:
  renderer:
    dpi: 200              # 渲染分辨率
    color_mode: "RGB"     # 颜色模式
    expand_margin: 16     # 裁剪扩边像素
  
  detector:
    min_figure_area: 5000 # 最小图片面积阈值
    confidence_threshold: 0.8

# Mistral API配置
mistral:
  model: "mistral-large-latest"
  temperature: 0.1        # 低温度确保稳定输出
  max_retries: 3

# 数据集配置
internvl2:
  task_weights:           # 各任务类型权重
    page_localization: 0.15
    figure_summary: 0.25
    variable_extraction: 0.15
    table_reading: 0.20
    multi_image_comparison: 0.10
    summary_qa: 0.15

# 质量控制
quality_control:
  consistency:
    strict_mode: true     # 严格模式
  deduplication:
    similarity_threshold: 0.95
```

## 常见问题

### Q1: 图片保存在哪里？
A: 处理后的图片保存在 `data/processed/[paper_id]/` 目录下，测试输出在 `output/test_extraction/`。

### Q2: 如何处理大批量PDF？
A: 将所有PDF放在 `data/raw_pdfs/` 目录，然后运行：
```bash
python scripts/run_enhanced_pipeline.py --pdf-dir data/raw_pdfs
```

### Q3: 检测效果不好怎么办？
A: 可以调整配置文件中的检测参数：
- 降低 `min_figure_area` 以检测更小的图片
- 调整 `expand_margin` 改变裁剪边距

### Q4: 如何只提取图片不做标注？
A: 使用 `--skip-annotation` 参数：
```bash
python scripts/run_enhanced_pipeline.py --skip-annotation
```

### Q5: 支持哪些PDF格式？
A: 支持标准PDF格式，包括扫描版PDF（如果图片已嵌入）。

## 开发说明

### 添加新的检测器
1. 在 `src/core/pdf_processor/` 创建新的检测器类
2. 继承基础检测接口
3. 实现 `detect_all_elements()` 方法

### 自定义标注模板
1. 修改 `src/dataset/qa_templates.py`
2. 添加新的问答模板
3. 在配置中调整任务权重

## 更新日志

- **当前版本**: 使用 `working_enhanced_detector.py` 作为主要检测器
- **已清理**: 移除旧版本代码，整理项目结构
- **优化**: 改进了图片提取精度和处理速度

## 许可证

[添加许可证信息]

## 联系方式

[添加联系信息]