# Check memory usage
Write-Host "=== TOP MEMORY CONSUMERS ===" -ForegroundColor Cyan
Write-Host ""

$totalRAM = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
Write-Host "Total RAM: $([math]::Round($totalRAM, 1)) GB"
Write-Host ""

Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 20 Name, @{N='Memory (MB)';E={[math]::Round($_.WorkingSet64/1MB,1)}}, CPU | Format-Table -AutoSize

Write-Host ""
Write-Host "=== KILLABLE (Non-Essential High Memory) ===" -ForegroundColor Yellow
$killable = @(
    "chrome", "msedge", "firefox", "Slack", "Discord", "Spotify",
    "Teams", "OneDrive", "Dropbox", "Steam", "EpicGames",
    "Code", "CodeHelper", "VisualStudio", "devenv",
    "Slack.exe", "Discord.exe", "Spotify.exe", "Teams.exe",
    "RuntimeBroker", "SearchHost", "ShellExperienceHost"
)
Get-Process | Where-Object { $killable -contains $_.Name -and $_.WorkingSet64/1MB -gt 100 } |
    Sort-Object WorkingSet64 -Descending |
    Select-Object Name, @{N='Memory (MB)';E={[math]::Round($_.WorkingSet64/1MB,1)}} |
    Format-Table -AutoSize
