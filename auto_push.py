#!/usr/bin/env python3
"""
自动推送到GitHub
注意：请不要在代码中硬编码密码！
"""
import subprocess
import sys
import getpass

def push_to_github():
    print("=== 自动推送 MedTuning 到 GitHub ===")
    print("仓库: https://github.com/yiranLou/Medtuning.git\n")
    
    # 获取用户凭据
    print("请输入你的GitHub凭据：")
    username = input("用户名 (yiranLou): ").strip() or "yiranLou"
    token = getpass.getpass("Personal Access Token: ")
    
    if not token:
        print("❌ 错误：需要 Personal Access Token")
        print("\n如何创建 Token:")
        print("1. 登录 GitHub")
        print("2. Settings → Developer settings → Personal access tokens")
        print("3. Generate new token (classic)")
        print("4. 勾选 'repo' 权限")
        print("5. 生成并复制 token")
        return False
    
    # 构建带认证的URL
    auth_url = f"https://{username}:{token}@github.com/yiranLou/Medtuning.git"
    
    try:
        # 临时设置远程URL（带认证）
        print("\n🔄 配置认证...")
        subprocess.run(["git", "remote", "set-url", "origin", auth_url], 
                      capture_output=True, text=True)
        
        # 执行推送
        print("🚀 推送中...")
        result = subprocess.run(["git", "push", "-u", "origin", "main"], 
                               capture_output=True, text=True)
        
        # 立即移除认证信息（安全考虑）
        subprocess.run(["git", "remote", "set-url", "origin", 
                       "https://github.com/yiranLou/Medtuning.git"], 
                      capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ 推送成功！")
            print(f"\n🎉 项目已推送到: https://github.com/{username}/Medtuning")
            print("\n📝 后续步骤：")
            print("   1. 访问仓库页面")
            print("   2. 添加项目描述")
            print("   3. 设置主题标签")
            return True
        else:
            print(f"\n❌ 推送失败！")
            print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        # 确保清理认证信息
        subprocess.run(["git", "remote", "set-url", "origin", 
                       "https://github.com/yiranLou/Medtuning.git"], 
                      capture_output=True, text=True)
        return False

if __name__ == "__main__":
    # 显示当前状态
    print("📊 当前提交历史:")
    subprocess.run(["git", "log", "--oneline", "-5"])
    print()
    
    # 执行推送
    success = push_to_github()
    sys.exit(0 if success else 1)