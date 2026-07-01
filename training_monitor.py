#!/usr/bin/env python3
"""
CARLA Grandmaster Training Monitor
Comprehensive real-time monitoring for autonomous driving training pipeline.

Features:
- GPU monitoring (nvidia-smi)
- CARLA server status (netstat port 2000)
- Training log parsing (last 10 lines)
- Python memory usage
- Thermal throttling detection
- Rich library support with ANSI fallback
- Telemetry logging to JSON

Usage:
    python training_monitor.py [--no-color]

Author: Claude
"""

import os
import sys
import json
import time
import socket
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# ============================================================================
# ANSI Color Codes (fallback when rich is not available)
# ============================================================================
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BOLD_RED = '\033[1m\033[91m'
    BOLD_YELLOW = '\033[1m\033[93m'
    BOLD_GREEN = '\033[1m\033[92m'
    BOLD_CYAN = '\033[1m\033[96m'

# ============================================================================
# Rich Library Support
# ============================================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'poll_interval': 10,  # seconds
    'log_file': r'D:\AI\traffic_vision\master_suite\training_output.log',
    'telemetry_dir': r'D:\AI\traffic_vision\master_suite\telemetry',
    'carla_port': 2000,
    'gpu_mem_warn_threshold': 11 * 1024,  # 11 GB in MB
    'gpu_util_warn_threshold': 95,  # percent
    'gpu_temp_warn_threshold': 83,  # degrees Celsius
    'training_freeze_threshold': 60,  # seconds
    'reward_collapse_threshold': -500,
}

# ============================================================================
# Color Helper Functions
# ============================================================================
def _strip_ansi(text: str) -> str:
    """Remove ANSI codes from text for length calculation."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def color(text: str, color_code: str) -> str:
    """Apply color to text."""
    if '--no-color' in sys.argv:
        return text
    return f"{color_code}{text}{Colors.RESET}"

def red(text: str) -> str:
    return color(text, Colors.RED)

def green(text: str) -> str:
    return color(text, Colors.GREEN)

def yellow(text: str) -> str:
    return color(text, Colors.YELLOW)

def cyan(text: str) -> str:
    return color(text, Colors.CYAN)

def bold(text: str) -> str:
    return color(text, Colors.BOLD)

def dim(text: str) -> str:
    return color(text, Colors.DIM)

def bold_red(text: str) -> str:
    return color(text, Colors.BOLD_RED)

def bold_green(text: str) -> str:
    return color(text, Colors.BOLD_GREEN)

def bold_yellow(text: str) -> str:
    return color(text, Colors.BOLD_YELLOW)

# ============================================================================
# GPU Monitoring (nvidia-smi)
# ============================================================================
def get_gpu_info() -> Dict[str, Any]:
    """Get GPU utilization, memory, temperature, and power using nvidia-smi."""
    gpu_info = {
        'name': 'Unknown',
        'utilization': 0,
        'memory_used': 0,
        'memory_total': 0,
        'memory_percent': 0,
        'temperature': 0,
        'power_draw': 0,
        'available': False,
        'error': None
    }

    try:
        # Query GPU info
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 6:
                gpu_info['name'] = parts[0].strip()
                gpu_info['utilization'] = int(parts[1].strip())
                gpu_info['memory_used'] = int(parts[2].strip())
                gpu_info['memory_total'] = int(parts[3].strip())
                gpu_info['temperature'] = int(parts[4].strip())
                gpu_info['power_draw'] = int(parts[5].strip())
                gpu_info['memory_percent'] = int(
                    (gpu_info['memory_used'] / gpu_info['memory_total'] * 100)
                    if gpu_info['memory_total'] > 0 else 0
                )
                gpu_info['available'] = True
    except FileNotFoundError:
        gpu_info['error'] = 'nvidia-smi not found'
    except subprocess.TimeoutExpired:
        gpu_info['error'] = 'nvidia-smi timeout'
    except Exception as e:
        gpu_info['error'] = str(e)

    return gpu_info

# ============================================================================
# CARLA Status Monitoring
# ============================================================================
def check_carla_status(port: int = 2000) -> Dict[str, Any]:
    """Check if CARLA is running and listening on the specified port."""
    status = {
        'running': False,
        'port_open': False,
        'status_text': 'STOPPED'
    }

    # Method 1: Check CARLA processes
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq CarlaUE4.exe'],
            capture_output=True, text=True, timeout=5
        )
        if 'CarlaUE4.exe' in result.stdout:
            status['running'] = True
    except Exception:
        pass

    # Method 2: Check port binding using netstat
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line.upper():
                status['port_open'] = True
                break
    except Exception:
        pass

    # Determine status text
    if status['running'] and status['port_open']:
        status['status_text'] = 'RUNNING'
    elif status['running']:
        status['status_text'] = 'RUNNING (port not bound)'
    elif status['port_open']:
        status['status_text'] = 'PORT OPEN (process unknown)'
    else:
        status['status_text'] = 'STOPPED'

    return status

# ============================================================================
# Training Log Monitoring
# ============================================================================
def read_training_log(log_file: str, lines: int = 10) -> Tuple[Dict[str, Any], List[str]]:
    """Read the last N lines of the training log file."""
    log_data = {
        'episode': 0,
        'steps': 0,
        'reward': 0,
        'loss': 0,
        'speed': 0,
        'throttle': 0,
        'status': 'UNKNOWN',
        'last_update': None
    }
    last_lines = []

    if not os.path.exists(log_file):
        return log_data, ['[Log file not found]']

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) >= lines else all_lines

            # Parse key metrics from log lines
            for line in reversed(last_lines):
                # Episode
                ep_match = re.search(r'[Ee]p(?:isode)?[:\s]*(\d+)', line)
                if ep_match:
                    log_data['episode'] = int(ep_match.group(1))

                # Steps
                step_match = re.search(r'[Ss]teps?[:\s]*(\d+)', line)
                if step_match:
                    log_data['steps'] = int(step_match.group(1))

                # Reward
                reward_match = re.search(r'[Rr]ew(?:ard)?[:\s]*(-?\d+\.?\d*)', line)
                if reward_match:
                    log_data['reward'] = float(reward_match.group(1))

                # Loss
                loss_match = re.search(r'[Ll]oss[:\s]*(\d+\.?\d*)', line)
                if loss_match:
                    log_data['loss'] = float(loss_match.group(1))

                # Speed
                speed_match = re.search(r'(\d+\.?\d*)\s*(?:km/?h|mph)', line, re.IGNORECASE)
                if speed_match:
                    log_data['speed'] = float(speed_match.group(1))

                # Throttle
                throttle_match = re.search(r'[Tt]hrottle[:\s]*(\d+\.?\d*)', line)
                if throttle_match:
                    log_data['throttle'] = float(throttle_match.group(1))

                # Status
                if 'active' in line.lower() or 'training' in line.lower():
                    log_data['status'] = 'ACTIVE'
                elif 'paused' in line.lower():
                    log_data['status'] = 'PAUSED'
                elif 'error' in line.lower() or 'exception' in line.lower():
                    log_data['status'] = 'ERROR'

            # Try to get timestamp from last line
            if last_lines:
                last_line = last_lines[-1]
                time_match = re.search(r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', last_line)
                if time_match:
                    log_data['last_update'] = time_match.group(1)

    except Exception as e:
        last_lines = [f'[Error reading log: {e}]']

    return log_data, last_lines

# ============================================================================
# Memory Monitoring
# ============================================================================
def get_python_memory() -> Dict[str, Any]:
    """Get current Python process memory usage."""
    mem_info = {
        'rss_mb': 0,
        'vms_mb': 0
    }

    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        mem_info['rss_mb'] = mem.rss / (1024 * 1024)
        mem_info['vms_mb'] = mem.vms / (1024 * 1024)
    except ImportError:
        # Fallback using os.popen
        try:
            result = subprocess.run(
                ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'WorkingSetSize'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                mem_info['rss_mb'] = int(lines[-1].strip()) / (1024 * 1024)
        except Exception:
            pass
    except Exception:
        pass

    return mem_info

# ============================================================================
# Alert System
# ============================================================================
class AlertManager:
    def __init__(self):
        self.alerts = []

    def check_alerts(self, gpu_info: Dict, carla_status: Dict,
                     log_data: Dict, last_update_time: Optional[float]) -> List[Tuple[str, str]]:
        """Check all alert conditions and return active alerts."""
        active_alerts = []
        now = time.time()

        # GPU Memory Starvation (> 11GB or > 95% util)
        if gpu_info['available']:
            if gpu_info['memory_used'] > CONFIG['gpu_mem_warn_threshold']:
                active_alerts.append(('GPU STARVATION WARNING',
                    f"GPU memory critical: {gpu_info['memory_used']} MB used"))
            elif gpu_info['utilization'] > CONFIG['gpu_util_warn_threshold']:
                active_alerts.append(('GPU UTILIZATION WARNING',
                    f"GPU at {gpu_info['utilization']}% utilization"))

        # Thermal Throttling (> 83°C)
        if gpu_info['available'] and gpu_info['temperature'] > CONFIG['gpu_temp_warn_threshold']:
            active_alerts.append(('THERMAL THROTTLING WARNING',
                f"GPU temperature: {gpu_info['temperature']}°C - throttling active"))

        # CARLA Crash
        if not carla_status['running'] and not carla_status['port_open']:
            active_alerts.append(('CARLA CRASHED - RESTARTING',
                "CARLA simulation not responding on port 2000"))

        # Training Frozen
        if last_update_time:
            time_since_update = now - last_update_time
            if time_since_update > CONFIG['training_freeze_threshold']:
                active_alerts.append(('TRAINING FROZEN - EMERGENCY RESCUE',
                    f"No log updates for {int(time_since_update)} seconds"))

        # Reward Collapse
        if log_data['reward'] < CONFIG['reward_collapse_threshold']:
            active_alerts.append(('REWARD COLLAPSE DETECTED',
                f"Reward dropped to {log_data['reward']}"))

        self.alerts = active_alerts
        return active_alerts

# ============================================================================
# Telemetry Logging
# ============================================================================
def save_telemetry(telemetry_dir: str, metrics: Dict[str, Any]) -> str:
    """Save current metrics to a JSON telemetry file."""
    os.makedirs(telemetry_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'metrics_{timestamp}.json'
    filepath = os.path.join(telemetry_dir, filename)

    # Add metadata
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['hostname'] = socket.gethostname()

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Failed to save telemetry: {e}")

    return filepath

# ============================================================================
# Display Formatters
# ============================================================================
def format_gpu_shortname(name: str) -> str:
    """Shorten GPU name for display."""
    if '4080' in name:
        return 'RTX 4080'
    elif '4090' in name:
        return 'RTX 4090'
    elif '3080' in name:
        return 'RTX 3080'
    elif '3090' in name:
        return 'RTX 3090'
    elif '3070' in name:
        return 'RTX 3070'
    elif 'A100' in name:
        return 'A100'
    elif 'A5000' in name:
        return 'A5000'
    elif 'V100' in name:
        return 'V100'
    return name[:12]

def format_gpu_display(gpu_info: Dict) -> str:
    """Format GPU information for display."""
    if not gpu_info['available']:
        return f"  GPU: {dim('N/A - nvidia-smi not available')}"

    name = format_gpu_shortname(gpu_info['name'])
    temp = gpu_info['temperature']
    temp_color = green if temp < 70 else yellow if temp < 83 else red

    vram = gpu_info['memory_used']
    vram_total = gpu_info['memory_total']
    vram_pct = gpu_info['memory_percent']

    util = gpu_info['utilization']
    util_color = green if util < 80 else yellow if util < 95 else red

    power = gpu_info['power_draw']

    return (f"  GPU: {bold(name)} | "
            f"VRAM: {vram}/{vram_total} MB ({vram_pct}%) | "
            f"{temp_color(str(temp) + '°C')} | "
            f"{util_color(str(util) + '%')} | "
            f"{power}W")

def format_carla_display(carla_status: Dict) -> str:
    """Format CARLA status for display."""
    status = carla_status['status_text']
    port = CONFIG['carla_port']

    if 'RUNNING' in status:
        status_color = green
    elif 'PORT' in status:
        status_color = yellow
    else:
        status_color = red

    return (f"  CARLA: {status_color(status)} | "
            f"Port {port}: {'LISTENING' if carla_status['port_open'] else dim('CLOSED')}")

def format_training_display(log_data: Dict) -> str:
    """Format training status for display."""
    status = log_data['status']
    status_color = green if status == 'ACTIVE' else yellow if status == 'PAUSED' else red

    ep = log_data['episode']
    steps = log_data['steps']
    reward = log_data['reward']

    return (f"  Training: {status_color(status)} | "
            f"Ep: {bold(str(ep))} | "
            f"Steps: {bold(str(steps))} | "
            f"Rew: {bold(str(int(reward)))}")

def format_system_display(log_data: Dict, gpu_info: Dict, python_mem: Dict) -> str:
    """Format overall system status."""
    # Determine overall status
    if log_data['status'] == 'ACTIVE' and gpu_info.get('available'):
        if gpu_info['temperature'] > 83 or gpu_info['utilization'] > 95:
            overall = 'DEGRADED'
            status_color = yellow
        else:
            overall = 'OPTIMAL'
            status_color = green
    elif log_data['status'] == 'ERROR' or not gpu_info.get('available'):
        overall = 'CRITICAL'
        status_color = red
    else:
        overall = 'UNKNOWN'
        status_color = dim

    speed = log_data.get('speed', 0)
    throttle = log_data.get('throttle', 0)

    # Add memory info if available
    mem_str = ""
    if python_mem['rss_mb'] > 0:
        mem_str = f" | RAM: {int(python_mem['rss_mb'])} MB"

    return (f"  Status: {status_color(overall)} | "
            f"Throttle: {throttle:.2f} | "
            f"Speed: {speed:.1f} km/h"
            f"{mem_str}")

def draw_box(lines: List[str], title: str = "CARLA GRANDMASTER TRAINING MONITOR") -> None:
    """Draw a text box around content using box-drawing characters."""
    width = 66
    divider = "═" * width

    print(f"╔{divider}╗")
    print(f"║{center_text(title, width)}║")
    print(f"╠{divider}╣")

    for line in lines:
        if line:
            # Calculate padding based on visible text length
            visible_len = len(_strip_ansi(line))
            padding = width - visible_len
            print(f"║{line}{' ' * padding}║")
        else:
            print(f"║{' ' * width}║")

    print(f"╚{divider}╝")

def center_text(text: str, width: int) -> str:
    """Center text within a width."""
    return text.center(width)

# ============================================================================
# Rich Display (when rich library is available)
# ============================================================================
def display_rich(gpu_info: Dict, carla_status: Dict, log_data: Dict,
                 python_mem: Dict, alerts: List[Tuple[str, str]],
                 last_lines: List[str], timestamp: str) -> None:
    """Display using rich library."""
    console.clear()

    # Create main content table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("metric", style="bold cyan", width=15)
    table.add_column("value", style="white")

    # GPU Row
    gpu_name = format_gpu_shortname(gpu_info['name']) if gpu_info['available'] else 'N/A'
    temp = gpu_info.get('temperature', 0)
    temp_color = "green" if temp < 70 else "yellow" if temp < 83 else "red bold"

    if gpu_info['available']:
        vram_pct = gpu_info['memory_percent']
        vram_color = "green" if vram_pct < 80 else "yellow" if vram_pct < 90 else "red bold"

        table.add_row(
            "[bold]GPU:[/bold]",
            f"[bold]{gpu_name}[/bold] | "
            f"[{vram_color}]VRAM: {gpu_info['memory_used']}/{gpu_info['memory_total']} MB ({vram_pct}%)[/{vram_color}] | "
            f"[{temp_color}]{temp}°C[/{temp_color}] | "
            f"{gpu_info['utilization']}% | {gpu_info['power_draw']}W"
        )
    else:
        table.add_row("[bold]GPU:[/bold]", "[dim]N/A - nvidia-smi not available[/dim]")

    # CARLA Row
    carla_ok = carla_status['running'] and carla_status['port_open']
    carla_style = "green" if carla_ok else "red bold"
    carla_status_text = "RUNNING" if carla_ok else carla_status['status_text']
    port_status = "[green]LISTENING[/green]" if carla_status['port_open'] else "[dim]CLOSED[/dim]"

    table.add_row(
        "[bold]CARLA:[/bold]",
        f"[{carla_style}]{carla_status_text}[/{carla_style}] | "
        f"Port {CONFIG['carla_port']}: {port_status}"
    )

    # Training Row
    train_style = "green" if log_data['status'] == 'ACTIVE' else "yellow" if log_data['status'] == 'PAUSED' else "red bold"
    table.add_row(
        "[bold]Training:[/bold]",
        f"[{train_style}]{log_data['status']}[/{train_style}] | "
        f"Ep: [bold]{log_data['episode']}[/bold] | "
        f"Steps: [bold]{log_data['steps']}[/bold] | "
        f"Rew: [bold]{int(log_data['reward'])}[/bold]"
    )

    # System Row
    speed = log_data.get('speed', 0)
    throttle = log_data.get('throttle', 0)
    ram = int(python_mem['rss_mb']) if python_mem['rss_mb'] > 0 else 0

    table.add_row(
        "[bold]Status:[/bold]",
        f"Throttle: {throttle:.2f} | Speed: {speed:.1f} km/h | RAM: {ram} MB"
    )

    # Print header
    print(f"\n[bold cyan]╔{'═' * 66}╗[/bold cyan]")
    print(f"[bold cyan]║{center_text('CARLA GRANDMASTER TRAINING MONITOR', 66)}║[/bold cyan]")
    print(f"[bold cyan]╠{'═' * 66}╣[/bold cyan]")
    print(f"[bold cyan]║{center_text(timestamp, 66)}║[/bold cyan]")
    print(f"[bold cyan]╚{'═' * 66}╝[/bold cyan]\n")

    console.print(table)

    # Alerts
    if alerts:
        print()
        for alert_text, alert_detail in alerts:
            console.print(f"[bold red]⚠ {alert_text}[/bold red]: {alert_detail}")

# ============================================================================
# Plain Display (fallback without rich)
# ============================================================================
def display_plain(gpu_info: Dict, carla_status: Dict, log_data: Dict,
                  python_mem: Dict, alerts: List[Tuple[str, str]],
                  last_lines: List[str], timestamp: str) -> None:
    """Display using plain print with ANSI colors."""
    os.system('cls' if os.name == 'nt' else 'clear')

    lines = [
        bold('CARLA GRANDMASTER TRAINING MONITOR') + ' - ' + dim(timestamp),
        "",
        format_gpu_display(gpu_info),
        format_carla_display(carla_status),
        format_training_display(log_data),
        format_system_display(log_data, gpu_info, python_mem),
        "",
        bold('Recent Log Output:'),
    ]

    # Add last few log lines
    for line in last_lines[-5:]:
        clean_line = line.strip()
        if clean_line:
            if 'error' in clean_line.lower() or 'exception' in clean_line.lower():
                lines.append(f"  {red(clean_line[:80])}")
            elif 'warn' in clean_line.lower():
                lines.append(f"  {yellow(clean_line[:80])}")
            else:
                lines.append(f"  {dim(clean_line[:80])}")

    draw_box(lines, "CARLA GRANDMASTER TRAINING MONITOR")

    # Alerts
    if alerts:
        print()
        for alert_text, alert_detail in alerts:
            print(f"{bold_red('⚠ ' + alert_text)}: {alert_detail}")

    print()
    print(f"{dim('Press Ctrl+C to exit | Next update in ' + str(CONFIG['poll_interval']) + 's...')}")

# ============================================================================
# Main Monitor Loop
# ============================================================================
def main():
    """Main monitoring loop."""
    alert_manager = AlertManager()
    last_log_update = None
    prev_log_data = {'steps': 0, 'reward': 0}
    iteration = 0
    last_telemetry_save = 0

    print(f"\n{'='*66}")
    print(f"  CARLA GRANDMASTER TRAINING MONITOR v1.0")
    print(f"{'='*66}")
    print(f"  Monitoring started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Poll interval: {CONFIG['poll_interval']}s")
    print(f"  Log file: {CONFIG['log_file']}")
    print(f"  Telemetry dir: {CONFIG['telemetry_dir']}")
    print(f"  Rich library: {'Available' if RICH_AVAILABLE else 'Not available - using ANSI fallback'}")
    print(f"{'='*66}\n")

    print("Initializing...")

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Collect all metrics
            gpu_info = get_gpu_info()
            carla_status = check_carla_status(CONFIG['carla_port'])
            log_data, last_lines = read_training_log(CONFIG['log_file'], lines=10)
            python_mem = get_python_memory()

            # Track log update time
            if log_data['steps'] != prev_log_data['steps']:
                last_log_update = time.time()
            prev_log_data = log_data.copy()

            # Check alerts
            alerts = alert_manager.check_alerts(
                gpu_info, carla_status, log_data, last_log_update
            )

            # Display
            if RICH_AVAILABLE:
                display_rich(gpu_info, carla_status, log_data, python_mem, alerts, last_lines, timestamp)
            else:
                display_plain(gpu_info, carla_status, log_data, python_mem, alerts, last_lines, timestamp)

            # Save telemetry every minute (approx 6 iterations)
            if iteration - last_telemetry_save >= 6:
                metrics = {
                    'iteration': iteration,
                    'gpu': gpu_info,
                    'carla': carla_status,
                    'training': log_data,
                    'memory': python_mem,
                    'alerts': [a[0] for a in alerts]
                }
                filepath = save_telemetry(CONFIG['telemetry_dir'], metrics)
                print(f"\n  {dim(f'Telemetry saved: {filepath}')}")
                last_telemetry_save = iteration

            # Wait for next poll
            time.sleep(CONFIG['poll_interval'])

    except KeyboardInterrupt:
        print(f"\n\n{bold_green('✓ Monitor stopped by user')}")
        print(f"Total iterations: {iteration}")
        print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"\n{bold_red('ERROR:')} Monitor crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
