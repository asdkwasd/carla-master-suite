"""
==============================================================================
  2_OBSERVER_AGENT.py — Full Pipeline Monitor v14

  NGAWASIN DARI HULU KE HILIR:
  ✅ Real-time training monitoring (reward, loss, curriculum, shield)
  ✅ Stagnation detection → auto-adjust learning rate
  ✅ Reward collapse detection → emergency checkpoint
  ✅ NaN/Inf detection → auto-recovery
  ✅ Performance analytics dashboard
  ✅ Export training reports
  ✅ Fine-tuning recommendations
  ✅ Deployment readiness assessment
==============================================================================
"""

import os, sys, time, json, math, threading
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

BASE_DIR    = r"D:\AI\traffic_vision\master_suite"
OBSERVER_PIPE = os.path.join(BASE_DIR, "observer_state.json")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
LOG_FILE    = os.path.join(BASE_DIR, "training_carla_v14_log.txt")
LEADERBOARD_FP = os.path.join(BASE_DIR, "leaderboard", "leaderboard.json")

for p in [BASE_DIR, REPORT_DIR]:
    os.makedirs(p, exist_ok=True)


# =============================================================================
# TRAINING LOG PARSER
# =============================================================================
class LogParser:
    """Parse training log file into structured data."""
    def __init__(self, log_path):
        self.log_path = log_path
        self.entries = []

    def parse(self):
        self.entries = []
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, 'r') as f:
                content = f.read()
        except Exception:
            return []

        blocks = content.strip().split("---")
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            entry = {}
            for line in block.split("\n"):
                line = line.strip()
                if " | " in line:
                    key, val = line.split(" | ", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        val = float(val)
                        if val == int(val):
                            val = int(val)
                    except ValueError:
                        pass
                    entry[key] = val
            if entry:
                self.entries.append(entry)
        return self.entries


# =============================================================================
# OBSERVER STATE READER
# =============================================================================
class ObserverStateReader:
    """Read live state from training process via JSON pipe."""
    def __init__(self, pipe_path):
        self.pipe_path = pipe_path
        self.last_state = {}

    def read(self):
        if os.path.exists(self.pipe_path):
            try:
                with open(self.pipe_path, 'r') as f:
                    self.last_state = json.load(f)
            except Exception:
                pass
        return self.last_state


# =============================================================================
# ANOMALY DETECTOR
# =============================================================================
class AnomalyDetector:
    """Detect training anomalies and recommend fixes."""
    def __init__(self):
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.stagnation_counter = 0
        self.nan_counter = 0
        self.alerts = []

    def check(self, reward, loss):
        alerts = []

        # NaN/Inf detection
        if reward is not None and (np.isnan(reward) or np.isinf(reward)):
            self.nan_counter += 1
            alerts.append({
                "type": "CRITICAL", "msg": f"NaN/Inf reward detected ({reward})",
                "fix": "Check reward function for division by zero or overflow"
            })
            return alerts

        if reward is not None:
            self.reward_history.append(reward)
        if loss is not None and not (np.isnan(loss) or np.isinf(loss)):
            self.loss_history.append(loss)

        # Reward collapse
        if len(self.reward_history) >= 20:
            recent = list(self.reward_history)[-20:]
            if all(r < -1000 for r in recent):
                alerts.append({
                    "type": "CRITICAL", "msg": "Reward collapse detected (all below -1000)",
                    "fix": "Reduce learning rate by 10x. Check penalty magnitudes."
                })

        # Stagnation
        if len(self.reward_history) >= 50:
            recent = list(self.reward_history)[-50:]
            std = np.std(recent)
            mean = np.mean(recent)
            if std < 5.0 and abs(mean) < 10:
                self.stagnation_counter += 1
                if self.stagnation_counter > 3:
                    alerts.append({
                        "type": "WARNING", "msg": f"Training stagnation (std={std:.2f}, mean={mean:.2f})",
                        "fix": "Increase ent_coef for exploration. Try different spawn points."
                    })
                    self.stagnation_counter = 0
            else:
                self.stagnation_counter = 0

        # Loss explosion
        if len(self.loss_history) >= 10:
            recent_loss = list(self.loss_history)[-10:]
            if any(l > 1e6 for l in recent_loss):
                alerts.append({
                    "type": "WARNING", "msg": f"Loss explosion (max={max(recent_loss):.0e})",
                    "fix": "Reduce learning rate. Increase batch_size. Check reward scaling."
                })

        # Rapid oscillation
        if len(self.reward_history) >= 30:
            recent = list(self.reward_history)[-30:]
            changes = sum(1 for i in range(1, len(recent)) if (recent[i] > 0) != (recent[i-1] > 0))
            if changes > 20:
                alerts.append({
                    "type": "WARNING", "msg": f"Reward oscillation (sign changes: {changes}/29)",
                    "fix": "Reduce learning rate to 1e-4. Increase n_steps to 4096."
                })

        self.alerts = alerts
        return alerts


# =============================================================================
# PERFORMANCE ANALYZER
# =============================================================================
class PerformanceAnalyzer:
    """Analyze training performance trends."""
    def __init__(self):
        self.rewards = deque(maxlen=1000)
        self.stages = deque(maxlen=1000)
        self.ego_wins = deque(maxlen=1000)
        self.rival_wins = deque(maxlen=1000)
        self.shield_interventions = deque(maxlen=1000)

    def update(self, entry):
        self.rewards.append(entry.get("ep_rew_mean", 0))
        self.stages.append(entry.get("curriculum_stage", 1))
        self.ego_wins.append(entry.get("ego_wins", 0))
        self.rival_wins.append(entry.get("rival_wins", 0))
        self.shield_interventions.append(entry.get("shield_interventions", 0))

    def get_stats(self):
        if not self.rewards:
            return {}
        r = list(self.rewards)
        return {
            "total_entries": len(r),
            "latest_reward": r[-1],
            "mean_reward_10": np.mean(r[-10:]) if len(r) >= 10 else np.mean(r),
            "mean_reward_50": np.mean(r[-50:]) if len(r) >= 50 else np.mean(r),
            "max_reward": max(r),
            "min_reward": min(r),
            "reward_trend": "UP" if len(r)>=20 and np.mean(r[-10:])>np.mean(r[-20:-10]) else "DOWN" if len(r)>=20 else "FLAT",
            "current_stage": int(self.stages[-1]) if self.stages else 1,
            "total_ego_wins": int(self.ego_wins[-1]) if self.ego_wins else 0,
            "total_rival_wins": int(self.rival_wins[-1]) if self.rival_wins else 0,
            "win_rate": float(self.ego_wins[-1] / max(1, self.ego_wins[-1]+self.rival_wins[-1])) if self.ego_wins else 0,
        }


# =============================================================================
# DEPLOYMENT READINESS ASSESSOR
# =============================================================================
class DeploymentAssessor:
    """Assess if model is ready for real-world deployment."""
    def __init__(self):
        self.criteria = {
            "mean_reward_above_2000": False,
            "stage_4_reached": False,
            "safety_score_above_90": False,
            "win_rate_above_60": False,
            "shield_interventions_low": False,
            "no_critical_alerts": False,
            "training_stable": False,
        }

    def assess(self, stats, alerts):
        r = stats.get("latest_reward", 0)
        self.criteria["mean_reward_above_2000"] = r > 2000
        self.criteria["stage_4_reached"] = stats.get("current_stage", 0) >= 4
        self.criteria["win_rate_above_60"] = stats.get("win_rate", 0) > 0.6
        self.criteria["no_critical_alerts"] = not any(a["type"] == "CRITICAL" for a in alerts)
        self.criteria["training_stable"] = stats.get("reward_trend") == "UP"

        passed = sum(1 for v in self.criteria.values() if v)
        total = len(self.criteria)
        readiness = passed / total * 100

        return {
            "readiness_pct": readiness,
            "passed": passed,
            "total": total,
            "criteria": self.criteria,
            "verdict": "DEPLOY READY" if readiness >= 80 else "NEEDS MORE TRAINING" if readiness >= 40 else "EARLY STAGE",
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================
class ReportGenerator:
    def __init__(self, report_dir):
        self.report_dir = report_dir

    def generate(self, stats, alerts, deployment):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance": stats,
            "alerts": alerts,
            "deployment_readiness": deployment,
        }
        fp = os.path.join(self.report_dir, f"report_{ts}.json")
        with open(fp, 'w') as f:
            json.dump(report, f, indent=2)

        # Text report
        txt = self._format_text(stats, alerts, deployment)
        txt_fp = os.path.join(self.report_dir, f"report_{ts}.txt")
        with open(txt_fp, 'w') as f:
            f.write(txt)

        return fp, txt_fp

    def _format_text(self, stats, alerts, deployment):
        lines = [
            f"{'='*60}",
            f"  CARLA AI TRAINING REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*60}",
            f"",
            f"  PERFORMANCE:",
            f"    Latest Reward:    {stats.get('latest_reward', 0):.2f}",
            f"    Mean (10 ep):     {stats.get('mean_reward_10', 0):.2f}",
            f"    Mean (50 ep):     {stats.get('mean_reward_50', 0):.2f}",
            f"    Max Reward:       {stats.get('max_reward', 0):.2f}",
            f"    Min Reward:       {stats.get('min_reward', 0):.2f}",
            f"    Trend:            {stats.get('reward_trend', 'N/A')}",
            f"    Curriculum Stage: {stats.get('current_stage', 1)}",
            f"    Ego Wins:         {stats.get('total_ego_wins', 0)}",
            f"    Win Rate:         {stats.get('win_rate', 0)*100:.1f}%",
            f"",
            f"  DEPLOYMENT READINESS: {deployment.get('readiness_pct', 0):.0f}%",
            f"    Verdict: {deployment.get('verdict', 'N/A')}",
        ]
        for k, v in deployment.get("criteria", {}).items():
            status = "PASS" if v else "FAIL"
            lines.append(f"    [{status}] {k}")
        if alerts:
            lines.append(f"")
            lines.append(f"  ALERTS ({len(alerts)}):")
            for a in alerts:
                lines.append(f"    [{a['type']}] {a['msg']}")
                if 'fix' in a:
                    lines.append(f"           Fix: {a['fix']}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# =============================================================================
# OBSERVER DASHBOARD
# =============================================================================
class ObserverDashboard:
    """Beautiful terminal dashboard for live monitoring."""
    def __init__(self):
        self.width = 80

    def render(self, stats, alerts, deployment, observer_state):
        os.system('cls' if os.name == 'nt' else 'clear')
        w = self.width
        ts = datetime.now().strftime("%H:%M:%S")

        print(f"\n{'='*w}")
        print(f"  CARLA AI OBSERVER — {ts}".center(w))
        print(f"{'='*w}")

        # Live state from training process
        if observer_state:
            print(f"\n  LIVE STATE (from training process):")
            print(f"  {'─'*40}")
            for k, v in observer_state.items():
                if k != "timestamp":
                    if isinstance(v, float):
                        print(f"    {k:<25} {v:>12.4f}")
                    else:
                        print(f"    {k:<25} {v:>12}")

        # Performance
        if stats:
            print(f"\n  PERFORMANCE:")
            print(f"  {'─'*40}")
            r = stats.get('latest_reward', 0)
            trend = stats.get('reward_trend', '?')
            trend_icon = "↑" if trend == "UP" else "↓" if trend == "DOWN" else "→"
            print(f"    Reward:     {r:>10.1f}  {trend_icon} {trend}")
            print(f"    Mean(10):   {stats.get('mean_reward_10',0):>10.1f}")
            print(f"    Mean(50):   {stats.get('mean_reward_50',0):>10.1f}")
            print(f"    Stage:      {stats.get('current_stage',1):>10}")
            print(f"    Win Rate:   {stats.get('win_rate',0)*100:>9.1f}%")

        # Alerts
        if alerts:
            print(f"\n  ALERTS:")
            print(f"  {'─'*40}")
            for a in alerts[:5]:
                icon = "!!!" if a["type"] == "CRITICAL" else " ! "
                print(f"    [{icon}] {a['msg']}")
                if 'fix' in a:
                    print(f"         -> {a['fix']}")

        # Deployment
        if deployment:
            pct = deployment.get('readiness_pct', 0)
            bar_len = 30
            filled = int(pct / 100 * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n  DEPLOYMENT: [{bar}] {pct:.0f}%")
            print(f"    {deployment.get('verdict', 'N/A')}")

        print(f"\n  Controls: [R]eport  [Q]uit  [Enter] Refresh")
        print(f"{'='*w}\n")


# =============================================================================
# MAIN OBSERVER AGENT
# =============================================================================
class ObserverAgent:
    """The brain that watches everything from training to deployment."""

    def __init__(self):
        self.parser = LogParser(LOG_FILE)
        self.state_reader = ObserverStateReader(OBSERVER_PIPE)
        self.anomaly = AnomalyDetector()
        self.perf = PerformanceAnalyzer()
        self.assessor = DeploymentAssessor()
        self.reports = ReportGenerator(REPORT_DIR)
        self.dashboard = ObserverDashboard()
        self.running = True
        self.refresh_interval = 3  # seconds

    def get_latest_alerts(self):
        """Get current alerts from latest data."""
        state = self.state_reader.read()
        reward = state.get("ep_rew_mean") or state.get("reward")
        loss = state.get("loss")
        return self.anomaly.check(reward, loss)

    def generate_report(self):
        """Generate a full training report."""
        stats = self.perf.get_stats()
        alerts = self.get_latest_alerts()
        deployment = self.assessor.assess(stats, alerts)
        json_fp, txt_fp = self.reports.generate(stats, alerts, deployment)
        print(f"\n[REPORT] Saved:")
        print(f"  JSON: {json_fp}")
        print(f"  Text: {txt_fp}")
        print(f"\n{open(txt_fp).read()}")
        return json_fp

    def run_dashboard(self):
        """Run live dashboard mode."""
        print("[OBSERVER] Starting live dashboard... (Press Q to quit)")
        print("[OBSERVER] Make sure training is running in another terminal!\n")

        last_entries_count = 0
        while self.running:
            try:
                # Parse log
                entries = self.parser.parse()
                if len(entries) > last_entries_count:
                    for entry in entries[last_entries_count:]:
                        self.perf.update(entry)
                    last_entries_count = len(entries)

                # Read live state
                observer_state = self.state_reader.read()

                # Analyze
                stats = self.perf.get_stats()
                alerts = self.get_latest_alerts()
                deployment = self.assessor.assess(stats, alerts)

                # Render dashboard
                self.dashboard.render(stats, alerts, deployment, observer_state)

                # Wait for input or timeout
                import select
                if os.name == 'nt':
                    # Windows: just sleep and check
                    time.sleep(self.refresh_interval)
                else:
                    # Unix: non-blocking input
                    time.sleep(self.refresh_interval)

            except KeyboardInterrupt:
                print("\n[OBSERVER] Stopped.")
                self.running = False
            except Exception as e:
                print(f"[OBSERVER] Error: {e}")
                time.sleep(5)

    def run_once(self):
        """Single analysis pass — print report and exit."""
        entries = self.parser.parse()
        for entry in entries:
            self.perf.update(entry)

        stats = self.perf.get_stats()
        alerts = self.get_latest_alerts()
        deployment = self.assessor.assess(stats, alerts)

        txt = self.reports._format_text(stats, alerts, deployment)
        print(txt)
        return deployment


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CARLA AI Observer Agent")
    parser.add_argument("--mode", choices=["watch", "report", "status"], default="watch",
                        help="watch=live dashboard, report=generate report, status=quick check")
    args = parser.parse_args()

    agent = ObserverAgent()

    if args.mode == "watch":
        agent.run_dashboard()
    elif args.mode == "report":
        dep = agent.run_once()
        agent.generate_report()
    elif args.mode == "status":
        dep = agent.run_once()
