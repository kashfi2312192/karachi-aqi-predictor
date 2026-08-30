@echo off
setlocal

REM ============================================================
REM PEARLS AQI PREDICTOR
REM DAILY AUTOMATION
REM ============================================================

REM ------------------------------------------------------------
REM PROJECT ROOT
REM ------------------------------------------------------------

cd /d C:\Users\m\Downloads\karachi-aqi-predictor


REM ------------------------------------------------------------
REM ACTIVATE VIRTUAL ENVIRONMENT
REM ------------------------------------------------------------

call venv\Scripts\activate.bat


REM ------------------------------------------------------------
REM RUN DAILY PIPELINE
REM ------------------------------------------------------------

echo ============================================================
echo PEARLS AQI PREDICTOR
echo DAILY AUTOMATION
echo ============================================================

echo.
echo Starting daily AQI pipeline...
echo.

python src\pipeline\run_daily.py


REM ------------------------------------------------------------
REM CHECK RESULT
REM ------------------------------------------------------------

if %ERRORLEVEL% EQU 0 (

    echo.
    echo ============================================================
    echo DAILY PIPELINE COMPLETED SUCCESSFULLY
    echo ============================================================
    echo.

) else (

    echo.
    echo ============================================================
    echo DAILY PIPELINE FAILED
    echo ============================================================
    echo.

)


REM ------------------------------------------------------------
REM KEEP WINDOW OPEN WHEN RUN MANUALLY
REM ------------------------------------------------------------

pause

endlocal