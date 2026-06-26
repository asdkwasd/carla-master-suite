# Check detailed memory breakdown
Write-Host "=== MEMORY USAGE REPORT ===" -ForegroundColor Cyan
Write-Host ""

$totalRAM = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
$usedRAM = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1KB - (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB
$usedPercent = ($usedRAM / $totalRAM) * 100

Write-Host "Total RAM: $([math]::Round($totalRAM, 1)) GB"
Write-Host "Used RAM:  $([math]::Round($usedRAM, 1)) GB ($([math]::Round($usedPercent, 1))%)"
Write-Host ""

# Group Chrome processes
Write-Host "=== CHROME PROCESSES ===" -ForegroundColor Yellow
$chrome = Get-Process chrome -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum
if ($chrome) {
    $chromeMB = [math]::Round($chrome.Sum / 1MB, 1)
    Write-Host "Chrome TOTAL: $chromeMB MB ($($chrome.Count) processes)"
    Get-Process chrome -ErrorAction SilentlyContinue |
        Sort-Object WorkingSet64 -Descending |
        Select-Object Name, @{N='Memory (MB)';E={[math]::Round($_.WorkingSet64/1MB,1)}} |
        Format-Table -AutoSize
}
Write-Host ""

# Claude processes
Write-Host "=== CLAUDE INSTANCES ===" -ForegroundColor Cyan
$claude = Get-Process claude -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum
if ($claude) {
    $claudeMB = [math]::Round($claude.Sum / 1MB, 1)
    Write-Host "Claude TOTAL: $claudeMB MB ($($claude.Count) processes)"
}
Write-Host ""

# High CPU processes (potential issues)
Write-Host "=== HIGH CPU PROCESSES (>50%) ===" -ForegroundColor Red
Get-Process | Where-Object { $_.CPU -gt 50 } |
    Sort-Object CPU -Descending |
    Select-Object Name, @{N='CPU (%)';E={[math]::Round($_.CPU,1)}},
                  @{N='Memory (MB)';E={[math]::Round($_.WorkingSet64/1MB,1)}} |
    Select-Object -First 10 |
    Format-Table -AutoSize

Write-Host ""
Write-Host "=== AVAILABLE ACTIONS ===" -ForegroundColor Green
Write-Host "[1] Kill ALL Chrome processes (free ~1.9 GB)"
Write-Host "[2] Kill other Chrome (keep 1 main tab)"
Write-Host "[3] Kill high-CPU bloatware (Cloudflare, Omen, etc)"
Write-Host "[4] Kill ALL non-essential (Chrome + bloatware)"
Write-Host "[5] Show what to kill for maximum free memory"
