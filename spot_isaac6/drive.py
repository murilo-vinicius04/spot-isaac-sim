# Drive Spot from a terminal, separate from the Isaac window. Start teleop.py first, then in any terminal on the same
# machine (plain python3, no Isaac needed):
#   python3 spot_isaac6/drive.py
# Hold W/S forward/back, A/D sideways, Q/E turn (arrows work too); space stops, R resets Spot, X quits.
# A terminal only sees key presses and their auto-repeats, never the release, so a key counts as held until its repeats
# stop for HOLD_S. That also swallows the press/release flicker that remote desktops send for a held key.
import curses, json, socket, time

PORT, HOLD_S, RATE_HZ = 9870, 0.6, 20
KEYS = {ord("w"): (1, 0, 0), curses.KEY_UP: (1, 0, 0), ord("s"): (-1, 0, 0), curses.KEY_DOWN: (-1, 0, 0),
        ord("a"): (0, 1, 0), curses.KEY_LEFT: (0, 1, 0), ord("d"): (0, -1, 0), curses.KEY_RIGHT: (0, -1, 0),
        ord("q"): (0, 0, 1), ord("e"): (0, 0, -1)}
SPEED = (0.5, 0.4, 0.5)          # m/s, m/s, rad/s: same defaults as teleop.py, inside what the policy trained on


def main(scr):
    curses.curs_set(0); scr.nodelay(True); scr.keypad(True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.setblocking(False)
    seen, status, t_status, reset = {}, {}, 0.0, False
    while True:
        now = time.time()
        while (k := scr.getch()) != -1:
            if k in KEYS: seen[k] = now
            elif k in (ord("r"), ord("R")): reset = True; seen.clear()
            elif k == ord(" "): seen.clear()
            elif k in (ord("x"), ord("X")): return
            elif k < 256 and chr(k).lower() != chr(k) and ord(chr(k).lower()) in KEYS: seen[ord(chr(k).lower())] = now
        seen = {k: t for k, t in seen.items() if now - t < HOLD_S}
        v = [0, 0, 0]
        for k in seen:
            for i in range(3): v[i] += KEYS[k][i]
        cmd = [max(-1, min(1, v[i])) * SPEED[i] for i in range(3)]
        sock.sendto(json.dumps({"cmd": cmd, "reset": reset}).encode(), ("127.0.0.1", PORT)); reset = False
        try:
            while True: status = json.loads(sock.recv(4096)); t_status = now
        except (BlockingIOError, ValueError): pass
        scr.erase()
        scr.addstr(0, 0, "Spot drive  |  hold W/S A/D Q/E (or arrows)  space stop  R reset  X quit")
        scr.addstr(2, 0, "command  vx %+.2f m/s  vy %+.2f m/s  wz %+.2f rad/s" % tuple(cmd))
        if now - t_status < 1.0:
            scr.addstr(3, 0, "spot     x %+.2f  y %+.2f  speed %.2f m/s   sim %.1f s   %s" % (
                status["x"], status["y"], status["speed"], status["t"], "FELL - press R" if status["fell"] else "ok"))
        else:
            scr.addstr(3, 0, "no reply from the sim on udp %d - is teleop.py running?" % PORT)
        scr.refresh(); time.sleep(1.0 / RATE_HZ)


curses.wrapper(main)
