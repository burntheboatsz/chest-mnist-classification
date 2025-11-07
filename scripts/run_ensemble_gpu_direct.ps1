# PowerShell script untuk training Ensemble Model dengan GPU
# Menggunakan conda environment pytorch-gpu (full path)

$PYTHON_PATH = "C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENSEMBLE MODEL TRAINING WITH GPU" -ForegroundColor Cyan
Write-Host "SimpleCNN + ResNet18 Combination" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
& $PYTHON_PATH -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

Write-Host ""
Write-Host "Starting ensemble training..." -ForegroundColor Green
Write-Host ""

# Navigate and run
Set-Location -Path "D:\vscode\improved-chestmnist\chest-mnist-classification"
& $PYTHON_PATH train_ensemble.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TRAINING COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
