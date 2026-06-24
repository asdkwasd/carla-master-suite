"""
==============================================================================
  CARLA COMPETITION VIEWER — Real-time Leaderboard & Analytics

  Fitur:
  - Live leaderboard dengan auto-refresh
  - Visualisasi performa agent per episode
  - Memory hotspot visualization
  - Battle history
==============================================================================
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.layout import Layout
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("[WARN] 'rich' library not installed. Install with: pip install rich")

BASE_DIR = r"D:\AI\traffic_vision\master_suite"
LEADERBOARD_FILE = os.path.join(BASE_DIR, "leaderboard", "leaderboard.json")
BATTLE_LOG = os.path.join(BASE_DIR, "battle_log.json")
MEMORY_DIR = os.path.join(BASE_DIR, "agent_memories")


class CompetitionViewer:
    """Viewer real-time untuk leaderboard dan statistik kompetisi"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.leaderboard_data = {}
        self.battle_history = []

    def load_data(self):
        """Load semua data kompetisi"""
        # Load leaderboard
        if os.path.exists(LEADERBOARD_FILE):
            with open(LEADERBOARD_FILE, 'r') as f:
                self.leaderboard_data = json.load(f)

        # Load battle history
        if os.path.exists(BATTLE_LOG):
            with open(BATTLE_LOG, 'r') as f:
                self.battle_history = json.load(f)

    def get_ranking(self) -> List[Tuple[int, str, dict]]:
        """Dapatkan ranking terurut"""
        sorted_data = sorted(
            self.leaderboard_data.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )
        return [(i+1, aid, data) for i, (aid, data) in enumerate(sorted_data)]

    def print_leaderboard(self, top_n: int = 15):
        """Print leaderboard dengan format yang bagus"""

        if not RICH_AVAILABLE:
            self._print_simple_leaderboard(top_n)
            return

        table = Table(title="🏆 CARLA AI COMPETITION LEADERBOARD 🏆", show_header=True,
                      header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Agent ID", style="green", width=20)
        table.add_column("Score", style="yellow", width=10)
        table.add_column("Distance", style="blue", width=12)
        table.add_column("Penalty", style="red", width=10)
        table.add_column("Collisions", style="red", width=10)
        table.add_column("Safety", style="green", width=8)

        ranking = self.get_ranking()
        for rank, aid, data in ranking[:top_n]:
            # Highlight top 3
            rank_style = ""
            if rank == 1:
                rank_style = "🥇 "
            elif rank == 2:
                rank_style = "🥈 "
            elif rank == 3:
                rank_style = "🥉 "

            table.add_row(
                f"{rank_style}{rank}",
                aid,
                f"{data.get('score', 0):.1f}",
                f"{data.get('total_distance', 0):.1f}m",
                f"{data.get('total_penalty', 0):.1f}",
                str(data.get('collisions', 0)),
                f"{data.get('safety_score', 100):.1f}%"
            )

        self.console.print(table)

    def _print_simple_leaderboard(self, top_n: int):
        """Fallback tanpa rich library"""
        print("\n" + "="*80)
        print("  CARLA AI COMPETITION LEADERBOARD".center(80))
        print("="*80)
        print(f"{'Rank':<8} {'Agent ID':<22} {'Score':<12} {'Distance':<15} {'Safety':<10}")
        print("-"*80)

        ranking = self.get_ranking()
        for rank, aid, data in ranking[:top_n]:
            medal = ""
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"

            print(f"{medal} {rank:<6} {aid:<22} {data.get('score', 0):<12.1f} "
                  f"{data.get('total_distance', 0):<15.1f}m {data.get('safety_score', 100):<10.1f}%")
        print("="*80 + "\n")

    def print_agent_details(self, agent_id: str):
        """Print detail performa satu agent"""
        if agent_id not in self.leaderboard_data:
            print(f"[ERROR] Agent {agent_id} not found in leaderboard")
            return

        data = self.leaderboard_data[agent_id]

        if RICH_AVAILABLE:
            panel = Panel(f"""
[bold cyan]Agent ID:[/bold cyan] {agent_id}

[bold yellow]Performance Metrics:[/bold yellow]
  • Total Distance: {data.get('total_distance', 0):.1f} meters
  • Total Penalty: {data.get('total_penalty', 0):.1f}
  • Net Score: {data.get('score', 0):.1f}

[bold red]Safety Record:[/bold red]
  • Collisions: {data.get('collisions', 0)}
  • Off-roads: {data.get('off_roads', 0)}
  • Red Light Violations: {data.get('red_light_violations', 0)}
  • Safety Score: {data.get('safety_score', 100):.1f}%

[bold green]Exploration:[/bold green]
  • Waypoints Visited: {data.get('waypoints_visited', 0)}
  • Episodes Completed: {data.get('episodes_completed', 0)}
  • Best Single Distance: {data.get('best_single_distance', 0):.1f}m

[bold blue]Last Updated:[/bold blue] {data.get('last_updated', 'N/A')}
            """, title=f"📊 Agent Profile: {agent_id}", border_style="cyan")
            self.console.print(panel)
        else:
            print(f"\n{'='*60}")
            print(f"  Agent Profile: {agent_id}")
            print(f"{'='*60}")
            print(f"\nPerformance Metrics:")
            print(f"  • Total Distance: {data.get('total_distance', 0):.1f} meters")
            print(f"  • Total Penalty: {data.get('total_penalty', 0):.1f}")
            print(f"  • Net Score: {data.get('score', 0):.1f}")
            print(f"\nSafety Record:")
            print(f"  • Collisions: {data.get('collisions', 0)}")
            print(f"  • Off-roads: {data.get('off_roads', 0)}")
            print(f"  • Safety Score: {data.get('safety_score', 100):.1f}%")
            print(f"\nExploration:")
            print(f"  • Waypoints Visited: {data.get('waypoints_visited', 0)}")
            print(f"  • Episodes Completed: {data.get('episodes_completed', 0)}")
            print(f"\nLast Updated: {data.get('last_updated', 'N/A')}\n")

    def print_battle_history(self, last_n: int = 10):
        """Print history pertandingan"""
        if not self.battle_history:
            print("[INFO] No battle history found")
            return

        recent_battles = self.battle_history[-last_n:]

        if RICH_AVAILABLE:
            table = Table(title="⚔️ Recent Battles ⚔️", show_header=True)
            table.add_column("Time", style="cyan", width=16)
            table.add_column("Match", style="white", width=30)
            table.add_column("Winner", style="green", width=20)
            table.add_column("Score Diff", style="yellow", width=10)

            for battle in recent_battles:
                timestamp = battle.get("timestamp", "")[:16]
                agent1 = battle["agent1"].get("agent_id", "Unknown")
                agent2 = battle["agent2"].get("agent_id", "Unknown")
                winner = battle.get("winner", "draw")
                score1 = battle["agent1"].get("score", 0)
                score2 = battle["agent2"].get("score", 0)
                diff = abs(score1 - score2)

                table.add_row(
                    timestamp,
                    f"{agent1} vs {agent2}",
                    winner if winner != "draw" else "Draw",
                    f"{diff:.1f}"
                )

            self.console.print(table)
        else:
            print("\n" + "="*80)
            print("  Recent Battles".center(80))
            print("="*80)
            for battle in recent_battles:
                agent1 = battle["agent1"].get("agent_id", "Unknown")
                agent2 = battle["agent2"].get("agent_id", "Unknown")
                winner = battle.get("winner", "draw")
                print(f"  {agent1} vs {agent2} → Winner: {winner}")
            print("="*80 + "\n")

    def print_memory_hotspots(self, agent_id: str):
        """Visualisasi hotspot kecelakaan dari memory agent"""
        memory_file = os.path.join(MEMORY_DIR, f"{agent_id}_memory.json")

        if not os.path.exists(memory_file):
            print(f"[INFO] No memory file found for {agent_id}")
            return

        with open(memory_file, 'r') as f:
            memory = json.load(f)

        hotspots = memory.get("collision_hotspots", [])
        offroad_zones = memory.get("offroad_zones", [])

        if RICH_AVAILABLE:
            panel = Panel(f"""
[bold red]Collision Hotspots:[/bold red] {len(hotspots)}
[bold yellow]Off-road Zones:[/bold yellow] {len(offroad_zones)}
[bold green]Successful Waypoints:[/bold green] {len(memory.get('successful_route', []))}

[bold cyan]Top 5 Dangerous Areas:[/bold cyan]
            """, title=f"🧠 Memory Analysis: {agent_id}", border_style="cyan")

            # Sort hotspots by count
            sorted_hotspots = sorted(hotspots, key=lambda x: x.get("count", 0), reverse=True)[:5]

            hotspot_text = ""
            for i, hs in enumerate(sorted_hotspots, 1):
                hotspot_text += f"  {i}. (x:{hs['x']:.1f}, y:{hs['y']:.1f}) - {hs['count']} crashes\n"

            self.console.print(panel)
            self.console.print(hotspot_text)
        else:
            print(f"\nMemory Analysis for {agent_id}:")
            print(f"  • Collision Hotspots: {len(hotspots)}")
            print(f"  • Off-road Zones: {len(offroad_zones)}")
            print(f"  • Successful Waypoints: {len(memory.get('successful_route', []))}")

            sorted_hotspots = sorted(hotspots, key=lambda x: x.get("count", 0), reverse=True)[:5]
            print("\n  Top 5 Dangerous Areas:")
            for i, hs in enumerate(sorted_hotspots, 1):
                print(f"    {i}. (x:{hs['x']:.1f}, y:{hs['y']:.1f}) - {hs['count']} crashes")

    def live_dashboard(self, refresh_interval: int = 5):
        """Dashboard real-time dengan auto-refresh"""
        if not RICH_AVAILABLE:
            print("[ERROR] Live dashboard requires 'rich' library")
            print("       Install with: pip install rich")
            return

        try:
            with Live(self._generate_dashboard(), refresh_per_second=1//refresh_interval) as live:
                while True:
                    self.load_data()
                    live.update(self._generate_dashboard())
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n[INFO] Live dashboard stopped")

    def _generate_dashboard(self):
        """Generate layout untuk live dashboard"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        layout["header"].update(Align.center(
            "🏎️  CARLA AI COMPETITION - LIVE DASHBOARD 🏎️",
            style="bold cyan"
        ))

        layout["body"].split_row(
            Layout(name="leaderboard"),
            Layout(name="stats")
        )

        # Leaderboard panel
        ranking = self.get_ranking()[:10]
        lb_text = "🏆 TOP 10 AGENTS\n\n"
        for rank, aid, data in ranking:
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            lb_text += f"{medal} {aid[:18]:<18} Score: {data.get('score', 0):>7.1f}\n"

        layout["leaderboard"].update(Panel(lb_text, title="Leaderboard", border_style="green"))

        # Stats panel
        total_agents = len(self.leaderboard_data)
        total_battles = len(self.battle_history)
        total_distance = sum(d.get('total_distance', 0) for d in self.leaderboard_data.values())
        total_collisions = sum(d.get('collisions', 0) for d in self.leaderboard_data.values())

        stats_text = f"""
📊 STATISTICS

Total Agents: {total_agents}
Total Battles: {total_battles}
Total Distance: {total_distance:.1f}m
Total Collisions: {total_collisions}

⚠️  SAFETY SUMMARY
Avg Safety Score: {np.mean([d.get('safety_score', 100) for d in self.leaderboard_data.values()]):.1f}%
Safe Drivers: {sum(1 for d in self.leaderboard_data.values() if d.get('safety_score', 100) > 80)}
Reckless Drivers: {sum(1 for d in self.leaderboard_data.values() if d.get('safety_score', 100) < 50)}
        """

        layout["stats"].update(Panel(stats_text, title="Statistics", border_style="yellow"))

        layout["footer"].update(Align.center(
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to exit",
            style="dim"
        ))

        return layout


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CARLA Competition Viewer")
    parser.add_argument("--mode", choices=["view", "live", "agent", "battles", "memory"],
                        default="view", help="View mode")
    parser.add_argument("--agent", type=str, help="Agent ID for detailed view")
    parser.add_argument("--top", type=int, default=15, help="Number of top agents to show")
    parser.add_argument("--refresh", type=int, default=5, help="Live dashboard refresh interval (seconds)")

    args = parser.parse_args()

    viewer = CompetitionViewer()
    viewer.load_data()

    if args.mode == "view":
        viewer.print_leaderboard(top_n=args.top)

    elif args.mode == "live":
        viewer.live_dashboard(refresh_interval=args.refresh)

    elif args.mode == "agent":
        if not args.agent:
            # Show top agent
            ranking = viewer.get_ranking()
            if ranking:
                args.agent = ranking[0][1]
                print(f"[AUTO] Showing top agent: {args.agent}")
        viewer.print_agent_details(args.agent)
        viewer.print_memory_hotspots(args.agent)

    elif args.mode == "battles":
        viewer.print_battle_history(last_n=args.top)

    elif args.mode == "memory":
        if not args.agent:
            ranking = viewer.get_ranking()
            if ranking:
                args.agent = ranking[0][1]
        viewer.print_memory_hotspots(args.agent)
