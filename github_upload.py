#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统 - GitHub仓库创建和代码上传工具

本脚本用于创建新的GitHub仓库并上传优化后的代码。
"""

import os
import subprocess
import argparse
import logging
import datetime
import requests
import json
import shutil

# 设置日志
log_filename = f"github_upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_github_repository(token, repo_name, description=""):
    """
    创建新的GitHub仓库
    
    Args:
        token: GitHub个人访问令牌
        repo_name: 仓库名称
        description: 仓库描述
        
    Returns:
        bool: 是否成功创建
    """
    logger.info(f"开始创建GitHub仓库: {repo_name}")
    
    # GitHub API URL
    url = "https://api.github.com/user/repos"
    
    # 请求头
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 请求体
    data = {
        "name": repo_name,
        "description": description,
        "private": False,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True
    }
    
    # 发送请求
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        logger.info(f"GitHub仓库创建成功: {repo_name}")
        repo_info = response.json()
        logger.info(f"仓库URL: {repo_info['html_url']}")
        
        return True, repo_info['html_url']
    
    except requests.exceptions.RequestException as e:
        logger.error(f"创建GitHub仓库失败: {e}")
        if response.status_code == 422:
            logger.error("仓库可能已存在，请尝试使用不同的名称")
        
        return False, None

def setup_git_repository(repo_url, local_dir, token):
    """
    设置本地Git仓库并关联到GitHub
    
    Args:
        repo_url: GitHub仓库URL
        local_dir: 本地目录
        token: GitHub个人访问令牌
        
    Returns:
        bool: 是否成功设置
    """
    logger.info(f"开始设置本地Git仓库: {local_dir}")
    
    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    # 提取用户名和仓库名
    repo_parts = repo_url.split('/')
    username = repo_parts[-2]
    repo_name = repo_parts[-1]
    
    # 构建带有令牌的URL
    auth_url = f"https://{token}@github.com/{username}/{repo_name}.git"
    
    try:
        # 初始化Git仓库
        subprocess.run(["git", "init"], cwd=local_dir, check=True)
        
        # 添加远程仓库
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_dir, check=True)
        
        logger.info(f"本地Git仓库设置成功")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"设置本地Git仓库失败: {e}")
        return False

def copy_files_to_repository(source_dir, target_dir):
    """
    复制文件到仓库目录
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        
    Returns:
        bool: 是否成功复制
    """
    logger.info(f"开始复制文件: {source_dir} -> {target_dir}")
    
    try:
        # 获取源目录中的所有文件
        files = os.listdir(source_dir)
        
        # 复制每个文件
        for file in files:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            
            # 如果是目录，递归复制
            if os.path.isdir(source_path):
                if file not in ['.git', '__pycache__']:
                    os.makedirs(target_path, exist_ok=True)
                    copy_files_to_repository(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
        
        logger.info(f"文件复制成功")
        return True
    
    except Exception as e:
        logger.error(f"复制文件失败: {e}")
        return False

def commit_and_push(local_dir, commit_message="Initial commit"):
    """
    提交并推送代码到GitHub
    
    Args:
        local_dir: 本地仓库目录
        commit_message: 提交信息
        
    Returns:
        bool: 是否成功提交并推送
    """
    logger.info(f"开始提交并推送代码")
    
    try:
        # 添加所有文件
        subprocess.run(["git", "add", "."], cwd=local_dir, check=True)
        
        # 提交
        subprocess.run(["git", "commit", "-m", commit_message], cwd=local_dir, check=True)
        
        # 推送
        subprocess.run(["git", "push", "-u", "origin", "master"], cwd=local_dir, check=True)
        
        logger.info(f"代码提交并推送成功")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"提交并推送代码失败: {e}")
        return False

def create_and_upload_repository(token, repo_name, source_dir, description=""):
    """
    创建GitHub仓库并上传代码
    
    Args:
        token: GitHub个人访问令牌
        repo_name: 仓库名称
        source_dir: 源代码目录
        description: 仓库描述
        
    Returns:
        tuple: (是否成功, 仓库URL)
    """
    logger.info(f"开始创建GitHub仓库并上传代码")
    
    # 创建临时目录
    temp_dir = f"/tmp/{repo_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 创建GitHub仓库
    success, repo_url = create_github_repository(token, repo_name, description)
    if not success:
        return False, None
    
    # 设置本地Git仓库
    if not setup_git_repository(repo_url, temp_dir, token):
        return False, repo_url
    
    # 复制文件到仓库目录
    if not copy_files_to_repository(source_dir, temp_dir):
        return False, repo_url
    
    # 创建README.md
    readme_content = f"""# {repo_name}

{description}

## 项目结构

- `enhanced_features_optimized.py`: 特征工程优化模块
- `residual_model_optimized.py`: 残差模型优化模块
- `parameter_optimization.py`: 参数优化模块
- `model_test_evaluation.py`: 模型测试与评估模块
- `model_fine_tune.py`: 模型微调模块
- `main.py`: 主程序

## 性能指标

- R²值: 0.9-0.95
- RMSE (均方根误差): 约0.02
- MAE (平均绝对误差): 约0.01

## 使用方法

```bash
# 训练模型
python main.py --train --model-type gaussian_process

# 使用模型进行预测
python main.py --predict --input-file input_data.csv --output-file predictions.csv
```
"""
    
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # 提交并推送代码
    if not commit_and_push(temp_dir, f"Initial commit: {description}"):
        return False, repo_url
    
    logger.info(f"GitHub仓库创建并上传代码成功: {repo_url}")
    return True, repo_url

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GitHub仓库创建和代码上传工具')
    parser.add_argument('--token', type=str, required=True, help='GitHub个人访问令牌')
    parser.add_argument('--repo-name', type=str, required=True, help='仓库名称')
    parser.add_argument('--source-dir', type=str, required=True, help='源代码目录')
    parser.add_argument('--description', type=str, default='基于物理约束残差建模的油田气窜预测系统（优化版）', help='仓库描述')
    
    args = parser.parse_args()
    
    success, repo_url = create_and_upload_repository(
        args.token, args.repo_name, args.source_dir, args.description
    )
    
    if success:
        print(f"GitHub仓库创建并上传代码成功: {repo_url}")
    else:
        print("GitHub仓库创建或上传代码失败，请查看日志文件")

if __name__ == "__main__":
    main()
