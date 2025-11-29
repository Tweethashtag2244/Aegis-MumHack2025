# AEGIS CNN Model Training - PowerShell Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AEGIS CNN Model Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set default paths
$REAL_DIR = "C:\Users\kshau\Downloads\archive\real"
$AI_DIR = "C:\Users\kshau\Downloads\archive\ai"
$OUTPUT = "models\artifact_detector.h5"

# Check if directories exist
if (-not (Test-Path $REAL_DIR)) {
    Write-Host "ERROR: Real videos directory not found: $REAL_DIR" -ForegroundColor Red
    Write-Host "Please update REAL_DIR in this script or provide --real-dir argument" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $AI_DIR)) {
    Write-Host "WARNING: AI videos directory not found: $AI_DIR" -ForegroundColor Yellow
    Write-Host "Please update AI_DIR in this script or provide --ai-dir argument" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Creating AI directory... You can add AI videos there later." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $AI_DIR -Force | Out-Null
}

# Create models directory if it doesn't exist
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" -Force | Out-Null
}

Write-Host "Real videos directory: $REAL_DIR" -ForegroundColor Green
Write-Host "AI videos directory: $AI_DIR" -ForegroundColor Green
Write-Host "Output model: $OUTPUT" -ForegroundColor Green
Write-Host ""

# Run training
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host ""

python train_cnn_model.py `
    --real-dir "$REAL_DIR" `
    --ai-dir "$AI_DIR" `
    --output "$OUTPUT" `
    --epochs 50 `
    --batch-size 32

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "Model saved to: $OUTPUT" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Training failed. Check error messages above." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

Read-Host "Press Enter to exit"

