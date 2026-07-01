# CARLA UE4 Process Monitor Script
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$processes = Get-Process | Where-Object {($_.ProcessName -like '*Carla*') -or ($_.ProcessName -like '*UE4*')} | Select-Object Id,ProcessName,CPU,WorkingSet
$port2000 = netstat -an | Select-String ":2000.*LISTENING"

Write-Host "=== CARLA MONITOR ==="
Write-Host "Timestamp: $timestamp"
Write-Host ""
Write-Host "Process Status:"
if ($processes) {
    $processes | Format-Table -AutoSize
} else {
    Write-Host "  No CARLA/UE4 processes found"
}
Write-Host ""
Write-Host "Port 2000 Status:"
if ($port2000) {
    Write-Host "  LISTENING - CARLA server is active"
} else {
    Write-Host "  NOT LISTENING - CARLA server not detected"
}
Write-Host "====================="
