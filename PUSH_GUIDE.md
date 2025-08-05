# 推送到 GitHub 指南

## 准备工作

1. **确保你有仓库的写权限**
   - 登录 GitHub
   - 确认你是 https://github.com/yiranLou/Medtuning 的所有者或协作者

2. **配置 Git 认证**
   
   选择以下方式之一：

   ### 方式1：使用 Personal Access Token (推荐)
   ```bash
   # 1. 在 GitHub 创建 token:
   # Settings > Developer settings > Personal access tokens > Generate new token
   # 勾选 repo 权限
   
   # 2. 使用 token 作为密码
   git push origin master
   # Username: your-github-username
   # Password: your-personal-access-token
   ```

   ### 方式2：使用 SSH
   ```bash
   # 1. 生成 SSH key（如果没有）
   ssh-keygen -t ed25519 -C "your-email@example.com"
   
   # 2. 添加到 GitHub
   cat ~/.ssh/id_ed25519.pub
   # 复制内容到 GitHub: Settings > SSH and GPG keys > New SSH key
   
   # 3. 更改远程 URL
   git remote set-url origin git@github.com:yiranLou/Medtuning.git
   ```

## 推送代码

```bash
# 查看当前状态
git status

# 查看远程仓库
git remote -v

# 推送到主分支
git push -u origin master

# 如果是第一次推送，可能需要：
git push --set-upstream origin master
```

## 常见问题

### 1. 认证失败
```
remote: Support for password authentication was removed
```
解决：使用 Personal Access Token 替代密码

### 2. 仓库不存在
```
remote: Repository not found
```
解决：
- 检查仓库 URL 是否正确
- 确认你有仓库访问权限
- 如果仓库不存在，先在 GitHub 创建

### 3. 推送被拒绝
```
! [rejected] master -> master (non-fast-forward)
```
解决：
```bash
# 先拉取远程更改
git pull origin master --rebase

# 解决冲突后再推送
git push origin master
```

## 创建新仓库（如果需要）

1. 登录 GitHub
2. 点击右上角 "+" > "New repository"
3. 输入仓库名：`Medtuning`
4. 选择 Public 或 Private
5. 不要初始化 README（我们已经有了）
6. 创建仓库
7. 按照页面提示推送现有代码

## 推送后的步骤

1. **设置仓库描述**
   - 在仓库页面点击齿轮图标
   - 添加描述：Medical Literature Multimodal Dataset Builder for InternVL2

2. **添加主题标签**
   - medical-imaging
   - dataset-generation
   - multimodal-learning
   - pdf-processing

3. **配置 Actions（可选）**
   - 启用 GitHub Actions 进行自动测试

4. **发布 Release（可选）**
   ```bash
   git tag -a v1.0.0 -m "Initial release"
   git push origin v1.0.0
   ```

## 完整推送命令序列

```bash
# 当前目录：/mnt/d/Buffer/Work_B/helpother/medtuning-master/new

# 1. 确认状态
git status
git log --oneline

# 2. 推送
git push -u origin master

# 3. 推送标签（如果有）
git push --tags
```

祝推送顺利！如有问题，请查看 GitHub 文档或联系仓库管理员。