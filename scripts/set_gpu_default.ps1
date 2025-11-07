# set_gpu_default.ps1
# Script untuk set GPU sebagai default PERMANENT

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETTING GPU AS DEFAULT FOR ALL PYTHON SCRIPTS" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$PYTORCH_GPU_PATH = "C:\Users\hnafi\miniconda3\envs\pytorch-gpu"

# 1. Set User Environment Variable
Write-Host "1. Setting User Environment Variable PYTORCH_GPU_PATH..." -ForegroundColor Green
[System.Environment]::SetEnvironmentVariable("PYTORCH_GPU_PATH", $PYTORCH_GPU_PATH, [System.EnvironmentVariableTarget]::User)
Write-Host "   ✅ PYTORCH_GPU_PATH = $PYTORCH_GPU_PATH" -ForegroundColor White

# 2. Set CUDA Environment Variables
Write-Host ""
Write-Host "2. Setting CUDA Environment Variables..." -ForegroundColor Green
[System.Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", [System.EnvironmentVariableTarget]::User)
Write-Host "   ✅ CUDA_VISIBLE_DEVICES = 0" -ForegroundColor White

# 3. Add to PATH (if not already there)
Write-Host ""
Write-Host "3. Adding pytorch-gpu to User PATH..." -ForegroundColor Green
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)
$scriptsPath = "$PYTORCH_GPU_PATH\Scripts"
$pythonPath = "$PYTORCH_GPU_PATH"

if ($currentPath -notlike "*$scriptsPath*") {
    $newPath = "$scriptsPath;$pythonPath;$currentPath"
    [System.Environment]::SetEnvironmentVariable("PATH", $newPath, [System.EnvironmentVariableTarget]::User)
    Write-Host "   ✅ Added to PATH: $scriptsPath" -ForegroundColor White
    Write-Host "   ✅ Added to PATH: $pythonPath" -ForegroundColor White
} else {
    Write-Host "   ℹ️  Already in PATH" -ForegroundColor Yellow
}

# 4. Create .condarc untuk default environment
Write-Host ""
Write-Host "4. Setting default conda environment..." -ForegroundColor Green
$condarcPath = "$env:USERPROFILE\.condarc"
$condarcContent = "auto_activate_base: false`nenv_prompt: '({name}) '`nchannels:`n  - pytorch`n  - nvidia`n  - defaults"
$condarcContent | Out-File -FilePath $condarcPath -Encoding UTF8 -Force
Write-Host "   Created/Updated .condarc" -ForegroundColor White

# 5. Verification
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  VERIFICATION" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Environment Variables Set:" -ForegroundColor Green
Write-Host "  PYTORCH_GPU_PATH = $([System.Environment]::GetEnvironmentVariable('PYTORCH_GPU_PATH', [System.EnvironmentVariableTarget]::User))" -ForegroundColor White
Write-Host "  CUDA_VISIBLE_DEVICES = $([System.Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES', [System.EnvironmentVariableTarget]::User))" -ForegroundColor White

Write-Host ""
Write-Host "Testing GPU with pytorch-gpu environment..." -ForegroundColor Yellow
& "$PYTORCH_GPU_PATH\python.exe" check_gpu.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 IMPORTANT NOTES:" -ForegroundColor Yellow
Write-Host "1. Restart PowerShell atau VS Code untuk apply environment variables" -ForegroundColor White
Write-Host "2. Gunakan script 'train_gpu.ps1' untuk training (otomatis pakai GPU)" -ForegroundColor White
Write-Host "3. Atau panggil: python train.py (akan error jika GPU tidak tersedia)" -ForegroundColor White
Write-Host ""
