#!/usr/bin/env python3
"""LLM reactor dashboard for the iCUE LINK pump LCD.

Outer ring: GPU load (left half ZOTAC, right half TUF).
Core: segments that spin faster with load and glow brighter with power.
Center: generation tokens/sec across vLLM servers, total GPU watts, GPU temps.
--demo simulates the data.
"""
import math
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from lcd import LinkLCD  # noqa: E402

FPS = 12
SS = 2                      # supersampling factor for smooth edges
SIZE = 480
S = SIZE * SS
DEMO = "--demo" in sys.argv

VLLM_PORTS = [18090, 18091]
ZOTAC_BUS = "00000000:01:00.0"
TUF_BUS = "00000000:03:00.0"
POWER_MAX = 1150.0          # both cards near their 575 W limits

BLUE = (61, 174, 233)
ORANGE = (233, 120, 61)
WHITE = (240, 240, 245)
DIM = (110, 115, 125)
TRACK = (38, 40, 48)

FONT_BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"


def font(path, size):
    return ImageFont.truetype(path, size * SS)


F_BIG = font(FONT_BOLD, 86)
F_UNIT = font(FONT_BOLD, 22)
F_MID = font(FONT_BOLD, 34)
F_SMALL = font(FONT, 20)
F_TINY = font(FONT_BOLD, 16)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.load = {ZOTAC_BUS: 0.0, TUF_BUS: 0.0}
        self.power = {ZOTAC_BUS: 0.0, TUF_BUS: 0.0}
        self.temp = {ZOTAC_BUS: 0, TUF_BUS: 0}
        self.tok_s = 0.0
        self.running = 0
        self._last_tokens = None
        self._last_time = None

    def poll_gpus(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        with self.lock:
            for line in out.strip().splitlines():
                bus, util, watts, temp = [x.strip() for x in line.split(",")]
                self.load[bus] = float(util) / 100.0
                self.power[bus] = float(watts)
                self.temp[bus] = int(temp)

    def poll_vllm(self):
        tokens = 0.0
        running = 0
        for port in VLLM_PORTS:
            try:
                text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2).read().decode()
            except OSError:
                continue
            for m in re.finditer(r"^vllm:generation_tokens_total\{[^}]*\} ([0-9.e+]+)$", text, re.M):
                tokens += float(m.group(1))
            for m in re.finditer(r"^vllm:num_requests_running\{[^}]*\} ([0-9.e+]+)$", text, re.M):
                running += int(float(m.group(1)))
        now = time.time()
        with self.lock:
            if self._last_tokens is not None and tokens >= self._last_tokens:
                rate = (tokens - self._last_tokens) / (now - self._last_time)
                self.tok_s = 0.6 * self.tok_s + 0.4 * rate
            self._last_tokens, self._last_time = tokens, now
            self.running = running

    def demo(self, t):
        with self.lock:
            busy = 0.5 + 0.5 * math.sin(t * 0.25)
            self.load[ZOTAC_BUS] = min(1.0, busy + random.uniform(-0.05, 0.05))
            self.load[TUF_BUS] = min(1.0, max(0.0, busy * 0.9 + random.uniform(-0.05, 0.05)))
            self.power[ZOTAC_BUS] = 25 + 420 * self.load[ZOTAC_BUS]
            self.power[TUF_BUS] = 30 + 410 * self.load[TUF_BUS]
            self.temp[ZOTAC_BUS] = int(35 + 20 * busy)
            self.temp[TUF_BUS] = int(42 + 30 * busy)
            self.tok_s = 0.8 * self.tok_s + 0.2 * (260 * busy if busy > 0.15 else 0)
            self.running = int(busy * 6)

    def snapshot(self):
        with self.lock:
            return dict(load=dict(self.load), power=dict(self.power), temp=dict(self.temp),
                        tok_s=self.tok_s, running=self.running)


def poller(stats):
    start = time.time()
    while True:
        try:
            if DEMO:
                stats.demo(time.time() - start)
            else:
                stats.poll_gpus()
                stats.poll_vllm()
        except Exception:
            pass
        time.sleep(1.0)


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(t):
    # cool cyan -> amber -> hot red as power rises
    stops = [(0.0, (40, 200, 255)), (0.55, (255, 170, 40)), (1.0, (255, 40, 30))]
    t = max(0.0, min(1.0, t))
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if t <= x1:
            return lerp_color(c0, c1, (t - x0) / (x1 - x0))
    return stops[-1][1]


def arc(draw, radius, width, start, end, color):
    c = S / 2
    box = (c - radius, c - radius, c + radius, c + radius)
    draw.arc(box, start, end, fill=color, width=width)


def render(snap, phase, shown):
    img = Image.new("RGB", (S, S), (6, 7, 12))
    total_w = snap["power"][ZOTAC_BUS] + snap["power"][TUF_BUS]
    heat = min(1.0, total_w / POWER_MAX)
    core = heat_color(heat)

    # Core glow, blurred so it looks like light rather than a disc
    # Drawn and blurred at low resolution, then scaled up: same look, a fraction of the CPU
    g = 96
    glow = Image.new("RGB", (g, g), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    r = int(g * (0.20 + 0.10 * heat))
    gd.ellipse((g / 2 - r, g / 2 - r, g / 2 + r, g / 2 + r), fill=lerp_color((0, 0, 0), core, 0.25 + 0.45 * heat))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=g / 24)).resize((S, S), Image.BILINEAR)
    img = Image.blend(img, glow, 0.9)
    d = ImageDraw.Draw(img)

    # Outer load ring: left half ZOTAC (bottom -> top), right half TUF (bottom -> top)
    ring_r = S / 2 - 18 * SS
    ring_w = 20 * SS
    arc(d, ring_r, ring_w, 95, 265, TRACK)
    arc(d, ring_r, ring_w, 275, 445, TRACK)
    zl, tl = shown["zotac"], shown["tuf"]
    if zl > 0.005:
        arc(d, ring_r, ring_w, 265 - 170 * zl, 265, BLUE)
    if tl > 0.005:
        arc(d, ring_r, ring_w, 275, 275 + 170 * tl, ORANGE)

    # Spinning reactor segments
    seg_r = S / 2 - 58 * SS
    n = 24
    for i in range(n):
        a0 = phase + i * 360 / n
        pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(math.radians(a0 * 2) + phase / 30))
        col = lerp_color((20, 22, 30), core, pulse * (0.35 + 0.65 * heat))
        arc(d, seg_r, 8 * SS, a0, a0 + 360 / n * 0.55, col)

    c = S / 2
    idle = snap["tok_s"] < 0.5 and snap["running"] == 0
    if idle:
        d.text((c, c - 20 * SS), "IDLE", font=F_MID, fill=DIM, anchor="mm")
    else:
        d.text((c, c - 36 * SS), f"{shown['tok']:.0f}", font=F_BIG, fill=WHITE, anchor="mm")
        d.text((c, c + 18 * SS), "TOKENS / SEC", font=F_UNIT, fill=core, anchor="mm")
    d.text((c, c + 62 * SS), f"{total_w:.0f} W", font=F_MID, fill=WHITE, anchor="mm")
    d.text((c - 62 * SS, c + 108 * SS), f"{snap['temp'][ZOTAC_BUS]}°", font=F_SMALL, fill=BLUE, anchor="mm")
    d.text((c + 62 * SS, c + 108 * SS), f"{snap['temp'][TUF_BUS]}°", font=F_SMALL, fill=ORANGE, anchor="mm")
    if snap["running"]:
        d.text((c, c + 108 * SS), f"{snap['running']} req", font=F_TINY, fill=DIM, anchor="mm")

    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main():
    stats = Stats()
    threading.Thread(target=poller, args=(stats,), daemon=True).start()
    lcd = LinkLCD()
    lcd.set_brightness(100)
    phase = 0.0
    shown = {"zotac": 0.0, "tuf": 0.0, "tok": 0.0}
    last = time.time()
    while True:
        now = time.time()
        dt = now - last
        last = now
        snap = stats.snapshot()
        # Ease displayed values toward the latest readings
        k = min(1.0, 4.0 * dt)
        shown["zotac"] += (snap["load"][ZOTAC_BUS] - shown["zotac"]) * k
        shown["tuf"] += (snap["load"][TUF_BUS] - shown["tuf"]) * k
        shown["tok"] += (snap["tok_s"] - shown["tok"]) * k
        avg_load = (shown["zotac"] + shown["tuf"]) / 2
        phase = (phase + dt * (12 + 260 * avg_load)) % 360
        try:
            lcd.send_image(render(snap, phase, shown), quality=85)
        except OSError:
            time.sleep(2)
            try:
                lcd.close()
            except OSError:
                pass
            lcd = LinkLCD()
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - now)))


if __name__ == "__main__":
    main()
