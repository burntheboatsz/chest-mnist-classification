# Quick Setup Script for GitHub Upload
# ChestMNIST Classification Project

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ChestMNIST Classification - GitHub Upload Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Git Installation
Write-Host "[1/7] Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "  ✓ Git installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Git not found!" -ForegroundColor Red
    Write-Host "  → Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# 2. Backup old README
Write-Host "`n[2/7] Preparing README file..." -ForegroundColor Yellow
if (Test-Path "README.md") {
    Move-Item "README.md" "README_old.md" -Force
    Write-Host "  ✓ Old README backed up to README_old.md" -ForegroundColor Green
}
if (Test-Path "README_GITHUB.md") {
    Move-Item "README_GITHUB.md" "README.md" -Force
    Write-Host "  ✓ README_GITHUB.md renamed to README.md" -ForegroundColor Green
} else {
    Write-Host "  ⚠ README_GITHUB.md not found, using existing README.md" -ForegroundColor Yellow
}

# 3. Clean temporary files
Write-Host "`n[3/7] Cleaning temporary files..." -ForegroundColor Yellow
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Write-Host "  ✓ Cleaned __pycache__ directories" -ForegroundColor Green

# 4. Initialize Git (if needed)
Write-Host "`n[4/7] Initializing Git repository..." -ForegroundColor Yellow
if (!(Test-Path ".git")) {
    git init
    Write-Host "  ✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "  ✓ Git repository already exists" -ForegroundColor Green
}

# 5. Check Git config
Write-Host "`n[5/7] Checking Git configuration..." -ForegroundColor Yellow
$userName = git config --global user.name
$userEmail = git config --global user.email

if ([string]::IsNullOrEmpty($userName) -or [string]::IsNullOrEmpty($userEmail)) {
    Write-Host "  ⚠ Git user not configured!" -ForegroundColor Yellow
    Write-Host ""
    $name = Read-Host "  Enter your name"
    $email = Read-Host "  Enter your email"
    
    git config --global user.name "$name"
    git config --global user.email "$email"
    Write-Host "  ✓ Git user configured" -ForegroundColor Green
} else {
    Write-Host "  ✓ Git user: $userName <$userEmail>" -ForegroundColor Green
}

# 6. Show files to be added
Write-Host "`n[6/7] Files to be uploaded:" -ForegroundColor Yellow
Write-Host "  Checking .gitignore rules..." -ForegroundColor Gray

$filesToAdd = git ls-files --others --exclude-standard
$fileCount = ($filesToAdd | Measure-Object).Count

if ($fileCount -gt 0) {
    Write-Host "  ✓ $fileCount files ready to add" -ForegroundColor Green
    Write-Host "    (Run 'git status' to see full list)" -ForegroundColor Gray
} else {
    Write-Host "  ℹ No new files to add (may be already committed)" -ForegroundColor Cyan
}

# 7. Show next steps
Write-Host "`n[7/7] Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ┌─────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │  READY TO UPLOAD TO GITHUB!                         │" -ForegroundColor Cyan
Write-Host "  └─────────────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Step 1: Create GitHub repository" -ForegroundColor White
Write-Host "    → Visit: https://github.com/new" -ForegroundColor Gray
Write-Host "    → Name: chestmnist-classification" -ForegroundColor Gray
Write-Host "    → DO NOT initialize with README/gitignore" -ForegroundColor Gray
Write-Host ""
Write-Host "  Step 2: Run these commands:" -ForegroundColor White
Write-Host "    git add ." -ForegroundColor Green
Write-Host "    git commit -m `"Initial commit: DenseNet121 92.13% accuracy`"" -ForegroundColor Green
Write-Host "    git remote add origin https://github.com/YOUR_USERNAME/chestmnist-classification.git" -ForegroundColor Green
Write-Host "    git branch -M main" -ForegroundColor Green
Write-Host "    git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "  ⚠ IMPORTANT: Replace YOUR_USERNAME with your GitHub username!" -ForegroundColor Yellow
Write-Host ""
Write-Host "  📖 Full Guide: See GITHUB_UPLOAD_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Offer to show git status
$showStatus = Read-Host "Show git status now? (y/n)"
if ($showStatus -eq 'y' -or $showStatus -eq 'Y') {
    Write-Host ""
    git status
}

Write-Host "`n✅ Setup completed! Ready to upload to GitHub." -ForegroundColor Green
Write-Host ""
