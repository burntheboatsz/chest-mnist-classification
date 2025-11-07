# Universal GPU Training Script
# Otomatis menggunakan pytorch-gpu environment untuk SEMUA training

$PYTHON_GPU = "C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe"
$SCRIPT_DIR = "D:\vscode\improved-chestmnist\chest-mnist-classification"

function Show-Header {
    param($Title)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Check-GPU {
    Write-Host ""
    Write-Host "Checking GPU Status..." -ForegroundColor Yellow
    & $PYTHON_GPU -c @"
import torch
print('✅ PyTorch Version:', torch.__version__)
print('✅ CUDA Available:', torch.cuda.is_available())
print('✅ CUDA Version:', torch.version.cuda)
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print('❌ GPU NOT DETECTED!')
    exit(1)
"@
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ ERROR: GPU tidak terdeteksi!" -ForegroundColor Red
        Write-Host "Pastikan environment pytorch-gpu sudah disetup dengan benar." -ForegroundColor Red
        exit 1
    }
}

# Main Menu
Show-Header "GPU TRAINING - ChestMNIST Classification"
Check-GPU

Write-Host ""
Write-Host "Pilih training yang ingin dijalankan:" -ForegroundColor Green
Write-Host "1. SimpleCNN (Best Model - 81-84%)" -ForegroundColor White
Write-Host "2. ResNet18 Transfer Learning (80%)" -ForegroundColor White
Write-Host "3. Ensemble (SimpleCNN + ResNet18 - Target 85-88%)" -ForegroundColor White
Write-Host "4. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Masukkan pilihan (1-4)"

Set-Location -Path $SCRIPT_DIR

switch ($choice) {
    "1" {
        Show-Header "Training SimpleCNN with GPU"
        & $PYTHON_GPU train.py
    }
    "2" {
        Show-Header "Training ResNet18 with GPU"
        & $PYTHON_GPU train_resnet.py
    }
    "3" {
        Show-Header "Training Ensemble (SimpleCNN + ResNet18) with GPU"
        & $PYTHON_GPU train_ensemble.py
    }
    "4" {
        Write-Host "Keluar..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "❌ Pilihan tidak valid!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Show-Header "TRAINING COMPLETED!"
Write-Host ""
