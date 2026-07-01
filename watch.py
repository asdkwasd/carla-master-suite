"""On-demand viewer. Buka pas mau cek; tutup window -> GPU balik ke training.
  python watch.py --live     # nempel ke CARLA yang lagi eval
  python watch.py --replay   # putar video rute terakhir + telemetri (0 GPU CARLA)
"""
import argparse, glob, json, os, sys

def latest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None

def show_telemetry():
    f = latest("telemetry/*.json")
    if not f:
        print("(belum ada telemetri)"); return
    d = json.load(open(f)); a = d.get("aggregate", {})
    print(f"[{d.get('run_id')}] DS={a.get('driving_score')} "
          f"RC={a.get('route_completion')} | next={d.get('next_action')}")
    for x in d.get("top_failures", [])[:5]:
        print("  FAIL:", x.get("scenario"), "->", x.get("hypothesis"))

def live():
    import carla, pygame, numpy as np
    show_telemetry()
    client = carla.Client("localhost", 2000); client.set_timeout(10.0)
    world = client.get_world()
    ego = next((a for a in world.get_actors().filter("vehicle.*")
                if a.attributes.get("role_name") == "hero"), None)
    if ego is None:
        print("Ego belum ada (eval belum jalan?)."); return
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "960"); bp.set_attribute("image_size_y", "540")
    tf = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-12))
    cam = world.spawn_actor(bp, tf, attach_to=ego)
    pygame.init(); screen = pygame.display.set_mode((960, 540)); buf = {}
    cam.listen(lambda img: buf.__setitem__("f",
        np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]))
    try:
        run = True
        while run:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: run = False
            if "f" in buf:
                screen.blit(pygame.surfarray.make_surface(buf["f"].swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
    finally:
        cam.stop(); cam.destroy(); pygame.quit()  # destroy sensor -> bebasin GPU viewer

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    if args.live:
        live()
    else:
        show_telemetry()
        v = latest("videos/*.mp4")
        if v:
            if sys.platform.startswith("win"): os.startfile(v)          # Windows
            elif sys.platform.startswith("linux"): os.system(f'xdg-open "{v}"')
            else: os.system(f'open "{v}"')
