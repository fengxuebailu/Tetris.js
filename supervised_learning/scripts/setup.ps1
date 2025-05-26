# 设置虚拟环境并安装依赖

# 检查是否已存在虚拟环境
if (Test-Path .\.venv) {
    Write-Host "发现已存在的虚拟环境，是否重新创建？(y/n)"
    $answer = Read-Host
    if ($answer -eq 'y') {
        Remove-Item -Recurse -Force .\.venv
    } else {
        Write-Host "将使用现有虚拟环境"
        .\.venv\Scripts\Activate.ps1
        python -m pip install -r requirements.txt
        exit
    }
}

# 创建新的虚拟环境
Write-Host "创建新的虚拟环境..."
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 更新pip
python -m pip install --upgrade pip

# 安装依赖
Write-Host "安装依赖包..."
python -m pip install -r requirements.txt

Write-Host "环境设置完成！"
Write-Host "使用 'start.py' 开始训练或评估模型"
