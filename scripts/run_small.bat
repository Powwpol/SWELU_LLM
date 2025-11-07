@echo off
REM Script pour lancer l'entraînement du petit modèle (Windows)

echo ======================================
echo MambaSWELU - Small Model Training
echo ======================================

REM Check if config exists
if not exist "configs\small_model.yaml" (
    echo Error: configs\small_model.yaml not found
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Warning: No virtual environment found
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Run training
python scripts\train_small.py --config configs\small_model.yaml

echo.
echo ======================================
echo Training completed!
echo ======================================
pause

