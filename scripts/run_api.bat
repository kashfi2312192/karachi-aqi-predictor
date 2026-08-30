@echo off
setlocal

REM ============================================================
REM PEARLS AQI PREDICTOR
REM PRODUCTION FLASK API
REM ============================================================

cd /d C:\Users\m\Downloads\karachi-aqi-predictor

call venv\Scripts\activate.bat

echo ============================================================
echo PEARLS AQI PREDICTOR
echo PRODUCTION FLASK API
echo ============================================================

echo.
echo Starting Waitress production server...
echo.

waitress-serve --host=0.0.0.0 --port=5000 src.api.app:app

endlocal