@echo off
setlocal enabledelayedexpansion

:: Script install.bat pour llm-bench
:: Simule la commande "make install" du Makefile

:: Vérifier que uv est disponible
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: uv not found. Install it from https://docs.astral.sh/uv/
    exit /b 1
)

:: Étape 1: Installer comme un outil uv
echo Installing llm-bench as uv tool...
call uv tool install . --reinstall --force
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install llm-bench as uv tool
    exit /b 1
)

:: Étape 2: Exécuter l'initialisation qui crée la configuration
echo Initializing llm-bench configuration...
call llm-bench init
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to initialize llm-bench configuration
    exit /b 1
)

echo Install complete! Run 'llm-bench' from anywhere.
endlocal
