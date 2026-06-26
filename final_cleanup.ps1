# Kill ALL non-essential with restart protection
Write-Host "=== FINAL MEMORY CLEANUP ===" -ForegroundColor Cyan
Write-Host ""

$beforeFree = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
Write-Host "Free Memory Before: $([math]::Round($beforeFree, 1)) MB"
Write-Host ""

# Processes to kill - order matters for restart protection
$killGroups = @(
    # Stop monitoring first
    @{Name="Taskmgr"; Action="stop"},
    # Kill Chrome (prevents new windows from opening)
    @{Name="chrome"; Action="kill"},
    # Kill bloatware
    @{Name="OmenCommandCenterBackground"; Action="kill"},
    @{Name="HP.OMEN.VadRedirectHelper"; Action="kill"},
    @{Name="Cloudflare WARP"; Action="kill"},
    @{Name="CloudflareWARP"; Action="kill"},
    @{Name="NVIDIA Overlay"; Action="kill"},
    @{Name="NVIDIA Share"; Action="kill"},
    @{Name="NVIDIA Container"; Action="kill"},
    # Extra Claude instances (keep main one)
    @{Name="claude"; Action="keepfirst"}
)

$killed = 0
$freed = 0

foreach ($group in $killGroups) {
    $procs = Get-Process -Name $group.Name -ErrorAction SilentlyContinue
    if ($procs) {
        foreach ($p in $procs) {
            try {
                $mem = [math]::Round($p.WorkingSet64 / 1MB, 1)

                if ($group.Action -eq "keepfirst") {
                    # Keep only first Claude process
                    if ($killed -gt 0) {
                        $p.Kill()
                        Write-Host "[KILLED] $($group.Name) PID=$($p.Id) - $mem MB" -ForegroundColor Yellow
                        $freed += $mem
                    }
                    $killed++
                } else {
                    $p.Kill() | Out-Null
                    Write-Host "[KILLED] $($group.Name) PID=$($p.Id) - $mem MB" -ForegroundColor Yellow
                    $freed += $mem
                }
            } catch {
                Write-Host "[WARN] $($group.Name): $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

Start-Sleep -Seconds 3

$afterFree = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB

Write-Host ""
Write-Host "=== RESULTS ===" -ForegroundColor Green
Write-Host "Memory Freed:  ~$([math]::Round($freed, 0)) MB"
Write-Host "Free Before:   $([math]::Round($beforeFree, 1)) MB"
Write-Host "Free After:    $([math]::Round($afterFree, 1)) MB"
Write-Host ""

Write-Host "=== CURRENT TOP PROCESSES ===" -ForegroundColor Cyan
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name, @{N='Memory (MB)';E={[math]::Round($_.WorkingSet64/1MB,1)}} | Format-Table -AutoSize

Write-Host ""
Write-Host "READY FOR CARLA TRAINING!" -ForegroundColor Green
