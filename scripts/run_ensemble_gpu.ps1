# PowerShell script untuk training Ensemble Model dengan GPU
# Menggunakan conda environment pytorch-gpu

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENSEMBLE MODEL TRAINING WITH GPU" -ForegroundColor Cyan
Write-Host "SimpleCNN + ResNet18 Combination" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location -Path "D:\vscode\improved-chestmnist\chest-mnist-classification"

# Activate conda environment dan jalankan training
Write-Host "Activating pytorch-gpu environment..." -ForegroundColor Yellow
conda activate pytorch-gpu

Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

Write-Host ""
Write-Host "Starting ensemble training with GPU..." -ForegroundColor Green
Write-Host ""
python train_ensemble.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TRAINING COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
