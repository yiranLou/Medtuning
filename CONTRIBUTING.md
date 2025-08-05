# 贡献指南

感谢您对 MedTuning 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

1. 在 [Issues](https://github.com/yiranLou/Medtuning/issues) 中搜索是否已有相似问题
2. 如果没有，创建新的 Issue，并提供：
   - 问题的详细描述
   - 复现步骤
   - 期望行为
   - 实际行为
   - 环境信息（Python版本、操作系统等）

### 提交代码

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature-name`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature-name`
5. 创建 Pull Request

### 代码规范

- 使用 Python 3.8+ 特性
- 遵循 PEP 8 代码风格
- 使用 Black 格式化代码
- 添加类型注解
- 编写单元测试
- 更新相关文档

### 提交信息格式

```
<type>: <subject>

<body>

<footer>
```

类型（type）：
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 重构
- test: 测试相关
- chore: 构建过程或辅助工具的变动

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yiranLou/Medtuning.git
cd Medtuning

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装 pre-commit hooks
pre-commit install
```

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_schemas.py

# 运行测试并生成覆盖率报告
pytest --cov=src tests/
```

## 文档

- 代码中使用清晰的注释
- 为新功能更新 README.md
- 在 docstring 中使用示例代码
- 保持配置文件的注释更新

## 发布流程

1. 更新版本号
2. 更新 CHANGELOG.md
3. 创建标签：`git tag v1.0.0`
4. 推送标签：`git push origin v1.0.0`

## 社区准则

- 尊重所有贡献者
- 保持讨论专业和友好
- 欢迎新手参与
- 耐心解答问题

## 需要帮助？

- 查看 [文档](README.md)
- 在 Issues 中提问
- 发送邮件至 contact@yiranlou.com

再次感谢您的贡献！