# PowerShell script untuk training Ensemble Model
# SimpleCNN + ResNet18 untuk performa maksimal

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENSEMBLE MODEL TRAINING" -ForegroundColor Cyan
Write-Host "SimpleCNN + ResNet18 Combination" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check CUDA availability
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
Write-Host ""

# Navigate to project directory
Set-Location -Path "D:\vscode\improved-chestmnist\chest-mnist-classification"

# Run ensemble training
Write-Host "Starting ensemble training..." -ForegroundColor Green
Write-Host ""
python train_ensemble.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TRAINING COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
