#!/bin/bash

echo "=== 推送 MedTuning 到 GitHub ==="
echo "仓库地址: https://github.com/yiranLou/Medtuning.git"
echo ""

# 检查是否有未提交的更改
if [[ -n $(git status -s) ]]; then
    echo "⚠️  检测到未提交的更改："
    git status -s
    echo ""
    read -p "是否要先提交这些更改？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "请输入提交信息: " commit_msg
        git commit -m "$commit_msg"
    fi
fi

# 显示当前状态
echo ""
echo "📊 当前状态："
git log --oneline -5
echo ""

# 推送
echo "🚀 开始推送到 GitHub..."
echo "请在提示时输入："
echo "  用户名: yiranLou"
echo "  密码: 你的 Personal Access Token (不是GitHub密码!)"
echo ""

# 执行推送
git push -u origin main

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo ""
    echo "🎉 你的项目现在可以在以下地址访问："
    echo "   https://github.com/yiranLou/Medtuning"
    echo ""
    echo "📝 后续步骤："
    echo "   1. 访问仓库页面添加描述"
    echo "   2. 添加主题标签: medical-imaging, dataset-generation, multimodal-learning"
    echo "   3. 可选: 创建 Release"
else
    echo ""
    echo "❌ 推送失败！"
    echo ""
    echo "🔧 可能的解决方案："
    echo "   1. 确保你有仓库的写权限"
    echo "   2. 创建 Personal Access Token:"
    echo "      - GitHub → Settings → Developer settings → Personal access tokens"
    echo "      - Generate new token (classic)"
    echo "      - 勾选 'repo' 权限"
    echo "      - 使用生成的 token 作为密码"
    echo "   3. 如果仓库不存在，先在 GitHub 创建仓库"
fi