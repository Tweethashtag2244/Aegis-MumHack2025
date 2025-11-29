@echo off
REM AEGIS CNN Model Training - Windows Batch Script

echo ========================================
echo AEGIS CNN Model Training
echo ========================================
echo.

REM Set default paths
set REAL_DIR=C:\Users\kshau\Downloads\archive\real
set AI_DIR=C:\Users\kshau\Downloads\archive\ai
set OUTPUT=models\artifact_detector.h5

REM Check if directories exist
if not exist "%REAL_DIR%" (
    echo ERROR: Real videos directory not found: %REAL_DIR%
    echo Please update REAL_DIR in this script or provide --real-dir argument
    pause
    exit /b 1
)

if not exist "%AI_DIR%" (
    echo WARNING: AI videos directory not found: %AI_DIR%
    echo Please update AI_DIR in this script or provide --ai-dir argument
    echo.
    echo Creating AI directory... You can add AI videos there later.
    mkdir "%AI_DIR%"
)

REM Create models directory if it doesn't exist
if not exist "models" mkdir models

echo Real videos directory: %REAL_DIR%
echo AI videos directory: %AI_DIR%
echo Output model: %OUTPUT%
echo.

REM Run training
python train_cnn_model.py ^
    --real-dir "%REAL_DIR%" ^
    --ai-dir "%AI_DIR%" ^
    --output "%OUTPUT%" ^
    --epochs 50 ^
    --batch-size 32

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Training completed successfully!
    echo Model saved to: %OUTPUT%
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Training failed. Check error messages above.
    echo ========================================
)

pause

