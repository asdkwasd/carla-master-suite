# Kill non-essential processes to free memory for training
Write-Host "=== KILLING NON-ESSENTIAL PROCESSES ===" -ForegroundColor Cyan
Write-Host ""

$totalBefore = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
Write-Host "Free Memory BEFORE: $([math]::Round($totalBefore, 1)) MB"
Write-Host ""

# Process to kill (non-essential)
$killList = @(
    # Chrome - big memory hog
    @{Name="chrome"; Reason="Memory: 2.3 GB - keep browser for later"}

    # HP Omen bloatware - extremely high CPU
    @{Name="OmenCommandCenterBackground"; Reason="CPU 102,000% - Omen gaming hub bloat"}
    @{Name="HP.OMEN.VadRedirectHelper"; Reason="CPU 161,000% - Omen bloatware"}

    # Cloudflare WARP - high CPU
    @{Name="Cloudflare WARP"; Reason="CPU 20,000% - VPN bloat"}

    # NVIDIA Overlay - unnecessary
    @{Name="NVIDIA Overlay"; Reason="Memory + CPU bloat"}
)

$killed = @()
foreach ($item in $killList) {
    $proc = Get-Process -Name $item.Name -ErrorAction SilentlyContinue
    if ($proc) {
        $mem = [math]::Round(($proc | Measure-Object WorkingSet64 -Sum).Sum / 1MB, 1)
        foreach ($p in $proc) {
            try {
                $p.Kill()
                $killed += "$($item.Name) ($mem MB) - $($item.Reason)"
                Write-Host "[KILLED] $($item.Name) - $mem MB" -ForegroundColor Yellow
            } catch {
                Write-Host "[FAILED] $($item.Name) - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

Start-Sleep -Seconds 2

$totalAfter = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
Write-Host ""
Write-Host "=== MEMORY FREED ===" -ForegroundColor Green
Write-Host "Free Memory AFTER:  $([math]::Round($totalAfter, 1)) MB"
Write-Host "Memory FREED:      $([math]::Round($totalAfter - $totalBefore, 1)) MB"
Write-Host ""

if ($killed.Count -gt 0) {
    Write-Host "=== KILLED PROCESSES ===" -ForegroundColor Cyan
    $killed | ForEach-Object { Write-Host $_ }
}
