# ============================================================
# PEARLS AQI PREDICTOR
# FLASK API WINDOWS STARTUP AUTOMATION
# ============================================================

$ProjectRoot = "C:\Users\m\Downloads\karachi-aqi-predictor"
$BatchFile = "$ProjectRoot\scripts\run_api.bat"

$TaskName = "Pearls AQI Predictor - Flask API"

Write-Host "============================================================"
Write-Host "PEARLS AQI PREDICTOR"
Write-Host "FLASK API TASK SCHEDULER SETUP"
Write-Host "============================================================"
Write-Host ""

# ------------------------------------------------------------
# REQUIRE ADMINISTRATOR
# ------------------------------------------------------------

$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()

$principal = New-Object Security.Principal.WindowsPrincipal(
    $currentIdentity
)

if (-not $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)) {

    Write-Host "ERROR: PowerShell must be run as Administrator."
    Write-Host ""
    Write-Host "Right-click PowerShell and select:"
    Write-Host "Run as administrator"
    Write-Host ""

    exit 1
}

Write-Host "Administrator privileges confirmed."
Write-Host ""

# ------------------------------------------------------------
# VALIDATE BATCH FILE
# ------------------------------------------------------------

if (-not (Test-Path $BatchFile)) {

    Write-Host "ERROR: API batch file not found:"
    Write-Host $BatchFile

    exit 1
}

Write-Host "API batch file found:"
Write-Host $BatchFile
Write-Host ""

# ------------------------------------------------------------
# REMOVE EXISTING TASK
# ------------------------------------------------------------

Write-Host "Checking for existing API task..."

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

$Trigger = New-ScheduledTaskTrigger `
    -AtStartup

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# ------------------------------------------------------------
# PRINCIPAL
# ------------------------------------------------------------

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$TaskPrincipal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Highest

# ------------------------------------------------------------
# CREATE TASK
# ------------------------------------------------------------

Write-Host ""
Write-Host "Creating API startup task..."
Write-Host ""

try {

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $TaskPrincipal `
        -Description "Starts the Pearls AQI Predictor Flask API using the Waitress production server at Windows startup." `
        -Force `
        -ErrorAction Stop

}
catch {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "API TASK CREATION FAILED"
    Write-Host "============================================================"
    Write-Host ""

    Write-Host "Error:"
    Write-Host $_.Exception.Message

    Write-Host ""

    exit 1
}

# ------------------------------------------------------------
# VERIFY TASK
# ------------------------------------------------------------

$createdTask = Get-ScheduledTask `
    -TaskName $TaskName `
    -ErrorAction SilentlyContinue

if (-not $createdTask) {

    Write-Host ""
    Write-Host "ERROR: Task registration returned but task could not be found."

    exit 1
}

# ------------------------------------------------------------
# SUCCESS
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "API STARTUP TASK CREATED SUCCESSFULLY"
Write-Host "============================================================"
Write-Host ""

Write-Host "Task:"
Write-Host $TaskName

Write-Host ""
Write-Host "Trigger:"
Write-Host "Windows startup"

Write-Host ""
Write-Host "API:"
Write-Host "Flask"
Write-Host "    ↓"
Write-Host "Waitress"
Write-Host "    ↓"
Write-Host "Port 5000"

Write-Host ""
Write-Host "============================================================"
Write-Host "PEARLS AQI API READY FOR AUTOMATIC STARTUP"
Write-Host "============================================================"