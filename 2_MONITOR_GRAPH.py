"""
==============================================================================
  CARLA GRANDMASTER AI SUITE v8.0 — 2_MONITOR_GRAPH.py
  Real-time live dashboard:
    • Episode Reward Mean (rolling 50-ep window)
    • Policy Loss & Value Loss
    • Curriculum Stage progression
    • Success Rate (collision-free episodes %)
    • LiDAR obstacle distance (min sector)
  Reads from: training_carla_v2_log.txt  (written by TrainingLogger callback)
  Updates every 2 seconds.
==============================================================================
"""
import re
import os
import time
import collections
import matplotlib
matplotlib.use("TkAgg")          # explicit backend for stability
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
LOG_PATH     = r"D:\AI\traffic_vision\master_suite\training_carla_v2_log.txt"
REFRESH_MS   = 2000            # animation refresh interval (ms)
MAX_POINTS   = 200             # max data points shown on graphs

# ── Colour scheme (dark theme) ────────────────────────────────────────────────
BG_COLOR      = "#0d1117"
PANEL_COLOR   = "#161b22"
GRID_COLOR    = "#21262d"
TEXT_COLOR    = "#e6edf3"
GREEN         = "#3fb950"
RED           = "#f85149"
BLUE          = "#58a6ff"
ORANGE        = "#d29922"
PURPLE        = "#bc8cff"

# ==============================================================================
# DATA PARSER
# ==============================================================================
def parse_log(path: str) -> dict:
    """
    Reads the structured log file produced by TrainingLogger.
    Returns dict of aligned lists: timesteps, rewards, losses, stages.
    """
    result = {
        "timesteps": [], "rewards": [], "losses": [],
        "stages": [],    "success_rate": [],
    }

    if not os.path.exists(path):
        return result

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        print(f"[MONITOR] Log read error: {e}")
        return result

    # Parse each metric independently (they may not always appear together)
    ts_vals  = [float(x) for x in re.findall(
        r"total_timesteps\s*\|\s*([-+\d.eE]+)", content)]
    rew_vals = [float(x) for x in re.findall(
        r"ep_rew_mean\s*\|\s*([-+\d.eE]+)", content)]
    loss_vals= [float(x) for x in re.findall(
        r"loss\s*\|\s*([-+\d.eE]+)", content)]
    stg_vals = [int(x)   for x in re.findall(
        r"curriculum_stage\s*\|\s*(\d+)", content)]

    # Align to shortest common length
    min_len = min(len(ts_vals), len(rew_vals), len(loss_vals), len(stg_vals))
    if min_len == 0:
        return result

    # Deduplicate on timestep
    seen    = set()
    indices = []
    for i, ts in enumerate(ts_vals[:min_len]):
        if ts not in seen:
            seen.add(ts)
            indices.append(i)

    result["timesteps"] = [ts_vals[i]   for i in indices][-MAX_POINTS:]
    result["rewards"]   = [rew_vals[i]  for i in indices][-MAX_POINTS:]
    result["losses"]    = [loss_vals[i] for i in indices][-MAX_POINTS:]
    result["stages"]    = [stg_vals[i]  for i in indices][-MAX_POINTS:]

    # Synthetic success rate: episodes where reward > 0 (positive outcome)
    # Computed as rolling 20-point fraction of positive rewards
    rewards = result["rewards"]
    sr = []
    window = 20
    for i in range(len(rewards)):
        chunk = rewards[max(0, i - window):i + 1]
        sr.append(100.0 * sum(r > 0 for r in chunk) / len(chunk))
    result["success_rate"] = sr

    return result

# ==============================================================================
# FIGURE SETUP
# ==============================================================================
fig = plt.figure(figsize=(14, 9), facecolor=BG_COLOR)
fig.canvas.manager.set_window_title("CARLA Grandmaster — Live Training Monitor")

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    hspace=0.50,
    wspace=0.30,
    top=0.90, bottom=0.07, left=0.07, right=0.97
)

ax_reward  = fig.add_subplot(gs[0, :])    # full width — reward
ax_loss    = fig.add_subplot(gs[1, 0])    # policy loss
ax_success = fig.add_subplot(gs[1, 1])    # success rate
ax_stage   = fig.add_subplot(gs[2, 0])    # curriculum stage
ax_info    = fig.add_subplot(gs[2, 1])    # text info panel

STAGE_NAMES = {1: "Basics", 2: "Rules", 3: "Traffic", 4: "Expert"}
STAGE_COLORS= {1: GREEN, 2: BLUE, 3: ORANGE, 4: RED}

def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.6, linewidth=0.5)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.96, "CARLA Grandmaster v8.0 — Live Training Dashboard",
         ha="center", va="top", color=TEXT_COLOR,
         fontsize=13, fontweight="bold")

# ==============================================================================
# ANIMATION CALLBACK
# ==============================================================================
_update_counter = [0]

def animate(_frame):
    _update_counter[0] += 1
    data = parse_log(LOG_PATH)

    if not data["timesteps"]:
        # No data yet — show waiting message
        for ax in (ax_reward, ax_loss, ax_success, ax_stage):
            ax.clear()
            _style_ax(ax, "Waiting for training data …")
        ax_info.clear()
        ax_info.set_facecolor(PANEL_COLOR)
        ax_info.text(0.5, 0.5, "Log not found.\nStart 1_TRAIN_MASTER.py first.",
                     ha="center", va="center", color=ORANGE,
                     fontsize=9, transform=ax_info.transAxes)
        return

    x    = data["timesteps"]
    rew  = data["rewards"]
    loss = data["losses"]
    stg  = data["stages"]
    sr   = data["success_rate"]
    n    = len(x)
    cur_stage = stg[-1] if stg else 1

    # ── Reward plot ───────────────────────────────────────────────────────────
    ax_reward.clear()
    _style_ax(ax_reward, "Episode Reward Mean  (rolling)", "Timesteps", "Reward")
    ax_reward.plot(x, rew, color=GREEN, linewidth=1.5, label="ep_rew_mean")
    ax_reward.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    # Rolling mean overlay
    if n >= 10:
        roll = np.convolve(rew, np.ones(10) / 10, mode="valid")
        ax_reward.plot(x[9:], roll, color=PURPLE, linewidth=2,
                       linestyle="--", label="rolling-10 mean")
    ax_reward.set_title(
        f"Episode Reward  │  Latest: {rew[-1]:+.1f}  │  Peak: {max(rew):+.1f}",
        color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=6
    )
    ax_reward.legend(fontsize=7, facecolor=PANEL_COLOR,
                     labelcolor=TEXT_COLOR, framealpha=0.7)

    # ── Loss plot ─────────────────────────────────────────────────────────────
    ax_loss.clear()
    _style_ax(ax_loss, f"Policy Loss  │  Latest: {loss[-1]:.2e}",
              "Timesteps", "Loss")
    ax_loss.plot(x, loss, color=RED, linewidth=1.2, marker="", label="loss")
    ax_loss.legend(fontsize=7, facecolor=PANEL_COLOR,
                   labelcolor=TEXT_COLOR, framealpha=0.7)

    # ── Success rate ──────────────────────────────────────────────────────────
    ax_success.clear()
    _style_ax(ax_success,
              f"Success Rate (positive reward) │ {sr[-1]:.1f}%",
              "Timesteps", "Success %")
    ax_success.fill_between(x, sr, alpha=0.25, color=BLUE)
    ax_success.plot(x, sr, color=BLUE, linewidth=1.5, label="success %")
    ax_success.set_ylim(0, 105)
    ax_success.legend(fontsize=7, facecolor=PANEL_COLOR,
                      labelcolor=TEXT_COLOR, framealpha=0.7)

    # ── Curriculum stage bar ──────────────────────────────────────────────────
    ax_stage.clear()
    _style_ax(ax_stage, "Curriculum Stage", "Timesteps", "Stage")
    colors = [STAGE_COLORS.get(s, GRID_COLOR) for s in stg]
    ax_stage.scatter(x, stg, c=colors, s=6, zorder=3, label="stage")
    ax_stage.set_yticks([1, 2, 3, 4])
    ax_stage.set_yticklabels(
        [f"{i} – {STAGE_NAMES[i]}" for i in [1, 2, 3, 4]],
        color=TEXT_COLOR, fontsize=7
    )
    ax_stage.set_ylim(0.5, 4.5)

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.clear()
    ax_info.set_facecolor(PANEL_COLOR)
    ax_info.axis("off")

    elapsed = _update_counter[0] * REFRESH_MS / 1000
    lines = [
        ("Timesteps",    f"{int(x[-1]):,}" if x else "0"),
        ("Reward (last)",f"{rew[-1]:+.2f}" if rew else "0"),
        ("Loss (last)",  f"{loss[-1]:.3e}" if loss else "0"),
        ("Stage",        f"{cur_stage} – {STAGE_NAMES.get(cur_stage, '?')}"),
        ("Success Rate", f"{sr[-1]:.1f}%" if sr else "0%"),
        ("Monitor uptime",f"{int(elapsed // 60)}m {int(elapsed % 60)}s"),
        ("Data points",  str(n)),
    ]
    y_pos = 0.92
    for label, value in lines:
        ax_info.text(0.05, y_pos, label + ":",
                     color=TEXT_COLOR, fontsize=8,
                     transform=ax_info.transAxes, fontweight="bold")
        ax_info.text(0.55, y_pos, value,
                     color=GREEN, fontsize=8,
                     transform=ax_info.transAxes)
        y_pos -= 0.13

    fig.canvas.draw_idle()

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  📊  CARLA Grandmaster — Live Training Monitor")
    print(f"  Reading from: {LOG_PATH}")
    print(f"  Refresh: every {REFRESH_MS // 1000}s  |  Ctrl+C to stop")
    print("=" * 55)

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=REFRESH_MS,
        cache_frame_data=False
    )
    plt.show()
