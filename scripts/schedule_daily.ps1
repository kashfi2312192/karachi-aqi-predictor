# ============================================================
# PEARLS AQI PREDICTOR
# WINDOWS DAILY AUTOMATION SCHEDULER
# ============================================================

$ProjectRoot = "C:\Users\m\Downloads\karachi-aqi-predictor"
$BatchFile = "$ProjectRoot\scripts\run_daily.bat"

$TaskName = "Pearls AQI Predictor - Daily Forecast"

Write-Host "============================================================"
Write-Host "PEARLS AQI PREDICTOR"
Write-Host "WINDOWS TASK SCHEDULER SETUP"
Write-Host "============================================================"
Write-Host ""

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

if (-not (Test-Path $BatchFile)) {
    Write-Host "ERROR: Daily batch file not found:"
    Write-Host $BatchFile
    exit 1
}

Write-Host "Batch file found:"
Write-Host $BatchFile
Write-Host ""

# ------------------------------------------------------------
# REMOVE EXISTING TASK
# ------------------------------------------------------------

Write-Host "Checking for existing scheduled task..."

$existingTask = Get-ScheduledTask `
    -TaskName $TaskName `
    -ErrorAction SilentlyContinue

if ($existingTask) {

    Write-Host "Existing task found."
    Write-Host "Removing old task..."

    Unregister-ScheduledTask `
        -TaskName $TaskName `
        -Confirm:$false

    Write-Host "Old task removed."
}

# ------------------------------------------------------------
# ACTION
# ------------------------------------------------------------

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatchFile`"" `
    -WorkingDirectory $ProjectRoot

# ------------------------------------------------------------
# TRIGGER
# ------------------------------------------------------------

# Runs every day at 12:00 PM
$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At 12:00PM

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# ------------------------------------------------------------
# CREATE TASK
# ------------------------------------------------------------

Write-Host ""
Write-Host "Creating scheduled task..."

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Runs the Pearls AQI Predictor daily pipeline and generates the latest 3-day Karachi AQI forecast."

# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "SCHEDULER CREATED SUCCESSFULLY"
Write-Host "============================================================"
Write-Host ""

Write-Host "Task:"
Write-Host $TaskName

Write-Host ""
Write-Host "Schedule:"
Write-Host "Every day at 12:00 PM"

Write-Host ""
Write-Host "Pipeline:"
Write-Host "Hopsworks"
Write-Host "    ↓"
Write-Host "Feature retrieval"
Write-Host "    ↓"
Write-Host "Model Registry"
Write-Host "    ↓"
Write-Host "3-day prediction"
Write-Host "    ↓"
Write-Host "latest_predictions.csv"
Write-Host "    ↓"
Write-Host "Flask API"
Write-Host "    ↓"
Write-Host "Streamlit Dashboard"

Write-Host ""
Write-Host "============================================================"
Write-Host "PEARLS AQI PREDICTOR READY FOR DAILY AUTOMATION"
Write-Host "============================================================"