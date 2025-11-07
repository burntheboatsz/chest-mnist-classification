# add_to_powershell_profile.ps1
# Script untuk menambahkan auto-activation ke PowerShell Profile

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETUP POWERSHELL PROFILE - AUTO GPU ENVIRONMENT" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path -Parent $profilePath

# Create profile directory if not exists
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    Write-Host "✅ Created profile directory: $profileDir" -ForegroundColor Green
}

# Create or update profile
$profileContent = @'
# Auto-activate pytorch-gpu environment for ML projects
$ML_PROJECT_PATH = "D:\vscode\improved-chestmnist"
$PYTORCH_GPU_PYTHON = "C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe"

# Check if we're in ML project directory
$currentDir = Get-Location
if ($currentDir.Path -like "$ML_PROJECT_PATH*") {
    Write-Host "🔥 ML Project detected! Using GPU environment (pytorch-gpu)" -ForegroundColor Yellow
    
    # Set alias untuk python yang langsung pakai GPU
    function python { & $PYTORCH_GPU_PYTHON $args }
    function pip { & "C:\Users\hnafi\miniconda3\envs\pytorch-gpu\Scripts\pip.exe" $args }
    
    # Set environment variable
    $env:PYTORCH_GPU_ACTIVE = "1"
    
    Write-Host "✅ GPU Python active: pytorch-gpu environment" -ForegroundColor Green
}
'@

# Backup existing profile if exists
if (Test-Path $profilePath) {
    $backupPath = "$profilePath.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item -Path $profilePath -Destination $backupPath -Force
    Write-Host "✅ Backup existing profile to: $backupPath" -ForegroundColor Green
    
    # Read existing content
    $existingContent = Get-Content -Path $profilePath -Raw
    
    # Check if our content already exists
    if ($existingContent -like "*pytorch-gpu*") {
        Write-Host "ℹ️  PyTorch GPU setup already exists in profile, updating..." -ForegroundColor Yellow
        # Remove old content and add new
        $existingContent = $existingContent -replace '(?ms)# Auto-activate pytorch-gpu.*?(?=\n# |\z)', ''
    }
    
    $finalContent = $existingContent + "`n`n" + $profileContent
} else {
    $finalContent = $profileContent
}

# Write to profile
$finalContent | Out-File -FilePath $profilePath -Encoding UTF8 -Force
Write-Host "✅ Updated PowerShell profile: $profilePath" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. RESTART PowerShell atau VS Code Terminal" -ForegroundColor White
Write-Host "2. Navigate ke project folder: cd D:\vscode\improved-chestmnist" -ForegroundColor White
Write-Host "3. Ketik 'python' akan otomatis gunakan GPU environment!" -ForegroundColor White
Write-Host ""
Write-Host "✅ Dari sekarang, setiap kali buka terminal di folder project," -ForegroundColor Green
Write-Host "   Python akan OTOMATIS menggunakan pytorch-gpu environment!" -ForegroundColor Green
Write-Host ""
