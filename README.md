# MedTuning - 医学文献多模态数据集构建工具

基于 Mistral Document AI 的医学文献 PDF 结构化数据集构建工具，专为 InternVL2 模型微调设计。

## 🌟 特性

- **两层Schema设计**：文档级和边界框级标注，支持整页问答和图表理解
- **零/低幻觉生成**：基于结构化字段的模板化Q/A生成
- **智能标注策略**：分批处理、防漂移、锚定文本
- **完整质量控制**：Schema验证、坐标检查、去重、一致性检查
- **灵活采样策略**：支持任务权重配置、论文平衡、质量优先
- **医学领域优化**：支持医学术语、单位标准化、实验数据提取

## 📋 系统要求

- Python 3.8+
- 4GB+ RAM
- Mistral API 密钥

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/yiranLou/Medtuning.git
cd Medtuning

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 方式1: 环境变量
export MISTRAL_API_KEY="your_api_key_here"

# 方式2: .env文件
cp .env.example .env
# 编辑.env文件，填入你的API密钥
```

### 3. 准备PDF文件

```bash
# 将PDF文件放入data/raw_pdfs/目录
cp /path/to/your/pdfs/*.pdf data/raw_pdfs/
```

### 4. 运行流水线

```bash
# 处理所有PDF
python scripts/run_pipeline.py

# 处理单个PDF
python scripts/run_pipeline.py -f data/raw_pdfs/paper.pdf

# 调试模式
python scripts/run_pipeline.py --debug
```

### 5. 验证数据集

```bash
# 验证生成的数据集
python scripts/validate_dataset.py data/outputs/internvl2_dataset.jsonl

# 生成验证报告
python scripts/validate_dataset.py data/outputs/internvl2_dataset.jsonl -o validation_report
```

## 📁 项目结构

```
.
├── src/
│   ├── core/                    # 核心模块
│   │   ├── schemas/            # 数据模型定义
│   │   ├── pdf_processor/      # PDF处理
│   │   └── validators/         # 数据验证
│   ├── annotation/             # Mistral标注
│   ├── dataset/                # 数据集生成
│   └── quality/                # 质量控制
├── configs/                    # 配置文件
│   ├── config.yaml            # 主配置
│   ├── schemas/               # JSON Schema
│   └── templates/             # Q/A模板
├── scripts/                   # 执行脚本
├── data/                      # 数据目录
│   ├── raw_pdfs/             # 原始PDF
│   ├── processed/            # 处理结果
│   └── outputs/              # 最终输出
└── requirements.txt          # 依赖列表
```

## 📊 数据集格式

生成的数据集符合InternVL2标准JSONL格式：

```json
{
  "id": "PMC12345_abstract_qa",
  "image": "processed/PMC12345/pages/page_001.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n这篇论文的主要研究发现是什么？"
    },
    {
      "from": "gpt",
      "value": "根据摘要，本研究的主要发现包括..."
    }
  ],
  "width": 1654,
  "height": 2339
}
```

### 支持的任务类型

1. **页面定位** (page_grounding): 定位文档元素位置
2. **图表摘要** (figure_caption): 描述图表内容
3. **变量提取** (variable_extraction): 提取图表变量和单位
4. **表格读取** (table_reading): 将表格转换为结构化数据
5. **多图对比** (multi_figure): 对比分析多个图表
6. **摘要问答** (abstract_qa): 基于论文内容的问答

## ⚙️ 配置说明

主配置文件 `configs/config.yaml` 包含：

```yaml
# PDF处理配置
pdf_processing:
  renderer:
    page_dpi: 200  # 页面渲染DPI
    crop_dpi: 300  # 裁剪图渲染DPI

# Mistral API配置
mistral:
  model: "mistral-large-latest"
  temperature: 0.1
  max_tokens: 4096

# 任务权重配置
internvl2:
  task_weights:
    page_grounding: 0.15
    figure_caption: 0.40      # 重点任务
    variable_extraction: 0.15
    table_reading: 0.15
    multi_figure: 0.10
    abstract_qa: 0.05
```

## 🔧 高级用法

### 自定义Q/A模板

编辑 `configs/templates/qa_templates.json` 添加新的问答模板：

```json
{
  "templates": {
    "custom_task": {
      "questions": ["你的问题模板"],
      "answer_builder": "custom_answer_function"
    }
  }
}
```

### 批量处理

```python
from src.annotation import DocumentAnnotator
from pathlib import Path

# 批量处理多个PDF
annotator = DocumentAnnotator()
for pdf in Path("data/raw_pdfs").glob("*.pdf"):
    annotation = await annotator.annotate_document(pdf)
```

## 📈 性能优化

- 使用 `--max-workers` 控制并发数
- 调整 `batch_size` 优化内存使用
- 启用 `--skip-detection` 跳过图表检测（用于纯文本PDF）

## 🐛 故障排除

### 常见问题

1. **Mistral API错误**
   - 检查API密钥是否正确
   - 确认网络连接正常
   - 查看API配额

2. **内存不足**
   - 降低DPI设置
   - 减少并发处理数
   - 分批处理大量PDF

3. **图表检测失败**
   - 安装PDFFigures2（可选）
   - 使用备用检测器

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发指南

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black src/
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- Mistral AI - 提供强大的Document AI能力
- InternVL2团队 - 优秀的多模态模型
- PDFFigures2 - 图表检测工具

## 📧 联系方式

- GitHub Issues: [提交问题](https://github.com/yiranLou/Medtuning/issues)
- Email: contact@yiranlou.com