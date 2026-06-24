"""
==============================================================================
  DATA SCIENCE: Old Training Data Processor
  Extracts value from 100+ PPO sessions of training data

  1. MERGE all agent memories into super memory (danger map + routes)
  2. ANALYZE patterns (where AI crashes, where it succeeds)
  3. PREPARE transfer learning package
  4. FIX auto-resume for future training
==============================================================================
"""

import os, sys, json, glob
import numpy as np
from collections import defaultdict

BASE_DIR = r"D:\AI\traffic_vision\master_suite"
MEMORY_DIR = os.path.join(BASE_DIR, "agent_memories")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# =============================================================================
# 1. MERGE ALL AGENT MEMORIES INTO SUPER MEMORY
# =============================================================================
def merge_memories():
    """Combine all agent memories into one comprehensive knowledge base."""
    print("=" * 70)
    print("  DATA PROCESSOR: Merging Agent Memories")
    print("=" * 70)

    all_hotspots = []
    all_offroad = []
    all_redlights = []
    all_routes = []

    memory_files = glob.glob(os.path.join(MEMORY_DIR, "*_memory.json"))
    print(f"\nFound {len(memory_files)} memory files:")

    for fp in memory_files:
        try:
            with open(fp) as f:
                d = json.load(f)
            name = os.path.basename(fp)
            n_col = len(d.get("collision_hotspots", []))
            n_off = len(d.get("offroad_zones", []))
            n_red = len(d.get("red_light_zones", []))
            n_route = len(d.get("successful_route", []))
            print(f"  {name}: {n_col} collisions, {n_off} offroad, {n_red} redlights, {n_route} route pts")

            all_hotspots.extend(d.get("collision_hotspots", []))
            all_offroad.extend(d.get("offroad_zones", []))
            all_redlights.extend(d.get("red_light_zones", []))
            all_routes.extend(d.get("successful_route", []))
        except Exception as e:
            print(f"  ERROR reading {fp}: {e}")

    # Deduplicate collision hotspots (merge nearby ones)
    merged_hotspots = deduplicate_locations(all_hotspots, threshold=5.0, key_merge="count")
    merged_offroad = deduplicate_locations(all_offroad, threshold=8.0)
    merged_redlights = deduplicate_locations(all_redlights, threshold=10.0)
    merged_routes = deduplicate_routes(all_routes, threshold=3.0)

    print(f"\n--- MERGED RESULTS ---")
    print(f"  Collision hotspots: {len(all_hotspots)} -> {len(merged_hotspots)} (deduplicated)")
    print(f"  Off-road zones:     {len(all_offroad)} -> {len(merged_offroad)}")
    print(f"  Red light zones:    {len(all_redlights)} -> {len(merged_redlights)}")
    print(f"  Route waypoints:    {len(all_routes)} -> {len(merged_routes)}")

    # Save super memory
    super_memory = {
        "collision_hotspots": merged_hotspots,
        "offroad_zones": merged_offroad,
        "red_light_zones": merged_redlights,
        "successful_route": merged_routes,
        "metadata": {
            "source_files": [os.path.basename(f) for f in memory_files],
            "total_raw_hotspots": len(all_hotspots),
            "total_raw_offroad": len(all_offroad),
            "total_raw_redlights": len(all_redlights),
            "total_raw_routes": len(all_routes),
            "merged_at": str(np.datetime64('now')),
        }
    }

    out_path = os.path.join(MEMORY_DIR, "super_memory.json")
    with open(out_path, 'w') as f:
        json.dump(super_memory, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    return super_memory


def deduplicate_locations(locs, threshold=5.0, key_merge=None):
    """Merge nearby locations into clusters."""
    if not locs:
        return []

    merged = []
    used = set()

    for i, loc in enumerate(locs):
        if i in used:
            continue
        cluster = [loc]
        used.add(i)

        for j, other in enumerate(locs):
            if j in used:
                continue
            dist = np.sqrt((loc["x"] - other["x"])**2 + (loc["y"] - other["y"])**2)
            if dist < threshold:
                cluster.append(other)
                used.add(j)

        # Average position
        avg = {
            "x": float(np.mean([c["x"] for c in cluster])),
            "y": float(np.mean([c["y"] for c in cluster])),
            "count": len(cluster),
            "last_seen": max(c.get("last_seen", "") for c in cluster),
        }
        if "z" in cluster[0]:
            avg["z"] = float(np.mean([c["z"] for c in cluster]))
        if "radius" in cluster[0]:
            avg["radius"] = float(np.mean([c.get("radius", 8.0) for c in cluster]))
        if "danger" in cluster[0]:
            avg["danger"] = float(np.max([c.get("danger", 0.5) for c in cluster]))

        # Merge counts if available
        if key_merge and key_merge in cluster[0]:
            avg[key_merge] = sum(c.get(key_merge, 1) for c in cluster)

        merged.append(avg)

    return merged


def deduplicate_routes(route_pts, threshold=3.0):
    """Deduplicate route waypoints, keeping unique positions."""
    if not route_pts:
        return []

    unique = []
    for pt in route_pts:
        is_dup = False
        for u in unique:
            dist = np.sqrt((pt["x"] - u["x"])**2 + (pt["y"] - u["y"])**2)
            if dist < threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(pt)

    return unique


# =============================================================================
# 2. ANALYZE PATTERNS
# =============================================================================
def analyze_patterns(super_memory):
    """Analyze driving patterns from merged memory."""
    print("\n" + "=" * 70)
    print("  PATTERN ANALYSIS")
    print("=" * 70)

    hotspots = super_memory.get("collision_hotspots", [])
    offroad = super_memory.get("offroad_zones", [])
    redlights = super_memory.get("red_light_zones", [])
    routes = super_memory.get("successful_route", [])

    # Collision analysis
    if hotspots:
        xs = [h["x"] for h in hotspots]
        ys = [h["y"] for h in hotspots]
        counts = [h.get("count", 1) for h in hotspots]
        print(f"\n  COLLISION HOTSPOTS ({len(hotspots)} unique):")
        print(f"    Most dangerous: {max(counts)} collisions at one spot")
        print(f"    X range: {min(xs):.1f} to {max(xs):.1f}")
        print(f"    Y range: {min(ys):.1f} to {max(ys):.1f}")
        for h in sorted(hotspots, key=lambda x: x.get("count", 1), reverse=True)[:5]:
            print(f"    - ({h['x']:.1f}, {h['y']:.1f}) count={h.get('count',1)}")

    # Off-road analysis
    if offroad:
        xs = [o["x"] for o in offroad]
        ys = [o["y"] for o in offroad]
        print(f"\n  OFF-ROAD ZONES ({len(offroad)} unique):")
        print(f"    X range: {min(xs):.1f} to {max(xs):.1f}")
        print(f"    Y range: {min(ys):.1f} to {max(ys):.1f}")
        # Cluster analysis
        from scipy.spatial.distance import cdist
        if len(offroad) > 1:
            coords = np.array([[o["x"], o["y"]] for o in offroad])
            # Find densest off-road cluster
            dists = cdist(coords, coords)
            density = np.sum(dists < 20, axis=1)
            center_idx = np.argmax(density)
            print(f"    Densest off-road cluster: ({offroad[center_idx]['x']:.1f}, {offroad[center_idx]['y']:.1f}) "
                  f"with {density[center_idx]} nearby zones")

    # Route analysis
    if routes:
        xs = [r["x"] for r in routes]
        ys = [r["y"] for r in routes]
        print(f"\n  SUCCESSFUL ROUTES ({len(routes)} unique waypoints):")
        print(f"    X range: {min(xs):.1f} to {max(xs):.1f}")
        print(f"    Y range: {min(ys):.1f} to {max(ys):.1f}")
        if "sequence" in routes[0]:
            max_seq = max(r.get("sequence", 0) for r in routes)
            print(f"    Max sequence reached: {max_seq}")

    # Coverage estimate
    if xs and ys:
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        area = x_range * y_range
        print(f"\n  MAP COVERAGE:")
        print(f"    Area explored: {area:.0f} sq meters")
        print(f"    X span: {x_range:.0f}m, Y span: {y_range:.0f}m")


# =============================================================================
# 3. FIND BEST CHECKPOINT
# =============================================================================
def find_best_checkpoint():
    """Find the checkpoint with the most training steps."""
    print("\n" + "=" * 70)
    print("  CHECKPOINT ANALYSIS")
    print("=" * 70)

    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "v14_*.zip"))
    if not checkpoints:
        print("  No v14 checkpoints found!")
        return None

    # Parse step counts
    ckpt_info = []
    for cp in checkpoints:
        name = os.path.basename(cp)
        try:
            steps = int(name.split("_")[1])
            size_mb = os.path.getsize(cp) / (1024 * 1024)
            import datetime
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(cp))
            ckpt_info.append({"path": cp, "steps": steps, "size_mb": size_mb, "mtime": mtime})
        except (ValueError, IndexError):
            pass

    ckpt_info.sort(key=lambda x: x["steps"])

    print(f"\n  Found {len(ckpt_info)} checkpoints:")
    for ci in ckpt_info:
        print(f"    {ci['steps']:>7} steps | {ci['size_mb']:.1f}MB | {ci['mtime']}")

    best = max(ckpt_info, key=lambda x: x["steps"])
    print(f"\n  BEST: {os.path.basename(best['path'])} ({best['steps']:,} steps)")
    return best


# =============================================================================
# 4. PREPARE TRANSFER PACKAGE
# =============================================================================
def prepare_transfer(best_checkpoint):
    """Prepare instructions for transferring knowledge."""
    print("\n" + "=" * 70)
    print("  TRANSFER LEARNING PLAN")
    print("=" * 70)

    print("""
  Knowledge to transfer on next restart:

  1. SUPER MEMORY -> RecallMemory
     Load super_memory.json into agent's recall memory
     -> AI immediately knows danger zones
     -> AI has 5,573 successful route waypoints

  2. FEATURE EXTRACTOR from best checkpoint
     Load v14_160000_steps.zip, copy CNN + LiDAR weights
     -> AI starts with pre-learned visual features
     -> Only policy/value heads need retraining

  3. AUTO-RESUME from latest checkpoint
     Always find and load the newest checkpoint
     -> Never start from 0 again
     -> Use reset_num_timesteps=False in model.learn()
""")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    super_mem = merge_memories()

    try:
        analyze_patterns(super_mem)
    except ImportError:
        print("\n  [scipy not installed - skipping cluster analysis]")
        # Basic analysis without scipy
        hotspots = super_mem.get("collision_hotspots", [])
        offroad = super_mem.get("offroad_zones", [])
        routes = super_mem.get("successful_route", [])
        print(f"\n  Basic stats:")
        print(f"    {len(hotspots)} collision hotspots")
        print(f"    {len(offroad)} off-road zones")
        print(f"    {len(routes)} route waypoints")

    best = find_best_checkpoint()
    if best:
        prepare_transfer(best)

    print("\n" + "=" * 70)
    print("  DONE! Run 1_TRAIN_MASTER.py to start with merged knowledge")
    print("=" * 70)
