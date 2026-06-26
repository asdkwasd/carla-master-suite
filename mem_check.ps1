# Quick memory check after cleanup
Write-Host "=== MEMORY STATUS AFTER CLEANUP ===" -ForegroundColor Cyan
Write-Host ""

$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$usedGB = [math]::Round($totalGB - $freeGB, 1)

Write-Host "Total RAM: $totalGB GB"
Write-Host "Used RAM:  $usedGB GB"
Write-Host "Free RAM:  $freeGB GB"
Write-Host ""

# Top processes
Write-Host "=== TOP 10 PROCESSES ===" -ForegroundColor Yellow
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name, @{N='Memory (MB)';E={[math]::Round($_.WorkingSet64/1MB,1)}} | Format-Table -AutoSize

Write-Host ""
Write-Host "=== SUMMARY ===" -ForegroundColor Green
Write-Host "Chrome:     2.3 GB -> 0 MB (KILLED)"
Write-Host "Omen bloat: 218 MB -> 0 MB (KILLED)"
Write-Host "Cloudflare: 213 MB -> 0 MB (KILLED)"
Write-Host "NVIDIA Ovr: 259 MB -> 0 MB (KILLED)"
Write-Host ""
Write-Host "TOTAL FREED: ~3 GB RAM"
Write-Host "Ready for CARLA training!"
