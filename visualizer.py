#!/usr/bin/env python3
"""Spotify Terminal Visualizer — real-time frequency bars from mic input + Spotify metadata."""

import argparse
import curses
import io
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import sounddevice as sd

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# Spotify integration
# ---------------------------------------------------------------------------

def load_env():
    """Load .env file from script directory if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def init_spotify():
    """Return an authenticated Spotify client or None."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        return None

    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPOTIPY_REDIRECT_URI", "https://127.0.0.1:8888/callback")

    if not client_id or not client_secret:
        return None

    try:
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-read-currently-playing user-read-playback-state",
            cache_path=os.path.join(cache_dir, ".spotify_cache"),
        ))
        sp.current_playback()
        return sp
    except Exception:
        return None


class SpotifyPoller(threading.Thread):
    """Polls Spotify for currently playing track."""

    def __init__(self, sp):
        super().__init__(daemon=True)
        self.sp = sp
        self.track_name = ""
        self.artist = ""
        self._api_progress_ms = 0
        self.duration_ms = 0
        self.is_playing = False
        self._last_poll_time = time.monotonic()
        self._stop = threading.Event()
        self._art_info = ("", None)

    @property
    def progress_ms(self):
        if self.is_playing:
            elapsed = (time.monotonic() - self._last_poll_time) * 1000
            return min(self._api_progress_ms + elapsed, self.duration_ms)
        return self._api_progress_ms

    @property
    def album_art(self):
        return self._art_info

    def run(self):
        prev_art_url = ""
        while not self._stop.is_set():
            try:
                pb = self.sp.current_playback()
                if pb and pb.get("item"):
                    item = pb["item"]
                    self.track_name = item.get("name", "")
                    artists = item.get("artists", [])
                    self.artist = ", ".join(a["name"] for a in artists) if artists else ""
                    self._api_progress_ms = pb.get("progress_ms", 0) or 0
                    self.duration_ms = item.get("duration_ms", 0) or 0
                    self.is_playing = pb.get("is_playing", False)
                    self._last_poll_time = time.monotonic()

                    images = item.get("album", {}).get("images", [])
                    art_url = ""
                    if len(images) >= 2:
                        art_url = images[1]["url"]
                    elif images:
                        art_url = images[0]["url"]

                    if art_url and art_url != prev_art_url:
                        prev_art_url = art_url
                        try:
                            data = urllib.request.urlopen(art_url, timeout=5).read()
                            self._art_info = (art_url, data)
                        except Exception:
                            self._art_info = (art_url, None)
                    elif not art_url:
                        self._art_info = ("", None)
                        prev_art_url = ""
                else:
                    self.track_name = ""
                    self.artist = ""
                    self.is_playing = False
                    self._art_info = ("", None)
                    prev_art_url = ""
            except Exception:
                pass
            self._stop.wait(3)

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

class AudioCapture:
    def __init__(self, device=None, samplerate=44100, blocksize=2048):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.buffer = np.zeros(blocksize, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self.lock:
            self.buffer = mono.copy()

    def start(self):
        self.stream = sd.InputStream(
            device=self.device, channels=1, samplerate=self.samplerate,
            blocksize=self.blocksize, dtype="float32", callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_buffer(self):
        with self.lock:
            return self.buffer.copy()


# ---------------------------------------------------------------------------
# FFT / frequency band processing
# ---------------------------------------------------------------------------

def compute_bands(audio, samplerate, num_bands):
    n = len(audio)
    if n == 0:
        return np.zeros(num_bands)

    windowed = audio * np.hanning(n)
    fft_data = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_data)
    freqs = np.fft.rfftfreq(n, 1.0 / samplerate)

    freq_min = 60.0
    freq_max = min(16000.0, samplerate / 2)
    band_edges = np.logspace(np.log10(freq_min), np.log10(freq_max), num_bands + 1)

    bands = np.zeros(num_bands)
    for i in range(num_bands):
        lo, hi = band_edges[i], band_edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            bands[i] = np.mean(magnitudes[mask])
    return bands


# ---------------------------------------------------------------------------
# Color palette extraction from album art
# ---------------------------------------------------------------------------

GALAXY_GRADIENT = [39, 51, 87, 129, 135, 171, 213, 255]


def _rgb_to_256(r, g, b):
    """Map RGB to xterm-256 color index."""
    ri = round(r / 255 * 5)
    gi = round(g / 255 * 5)
    bi = round(b / 255 * 5)
    return 16 + 36 * ri + 6 * gi + bi


def extract_palette(img_data, n=8):
    """Extract n dominant colors from album art, sorted dark to bright."""
    if not HAS_PIL or not img_data:
        return None
    try:
        img = Image.open(io.BytesIO(img_data))
        img = img.resize((80, 80), Image.LANCZOS)
        img = img.convert("RGB")
        quantized = img.quantize(colors=max(n * 2, 16), method=Image.MEDIANCUT)
        pal = quantized.getpalette()
        colors = []
        for i in range(min(n * 2, 16)):
            r, g, b = pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]
            # Boost saturation to counteract quantization washing
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            r = max(0, min(255, int(gray + (r - gray) * 1.5)))
            g = max(0, min(255, int(gray + (g - gray) * 1.5)))
            b = max(0, min(255, int(gray + (b - gray) * 1.5)))
            lum = (r * 299 + g * 587 + b * 114) / 255000
            if lum > 0.08:
                colors.append((lum, r, g, b))
        if not colors:
            return None
        colors.sort(key=lambda c: c[0])
        if len(colors) >= n:
            step = len(colors) / n
            picked = [colors[int(i * step)] for i in range(n)]
        else:
            picked = list(colors)
            while len(picked) < n:
                picked.append(picked[-1])
        return [_rgb_to_256(r, g, b) for _, r, g, b in picked]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Galaxy theme — colors & rendering
# ---------------------------------------------------------------------------

# Color pairs:
#   1-8:  bar gradient (galaxy default or album palette)
#   10: cyan (text)    11: purple (sep)
#   12: pink (✦ accent)   13: cyan (progress) 14: dark gray (dim)

def init_colors():
    curses.start_color()
    curses.use_default_colors()

    if curses.COLORS >= 256:
        for i, c in enumerate(GALAXY_GRADIENT):
            curses.init_pair(i + 1, c, -1)
        curses.init_pair(10, 51, -1)
        curses.init_pair(11, 99, -1)
        curses.init_pair(12, 213, -1)
        curses.init_pair(13, 51, -1)
        curses.init_pair(14, 238, -1)
    else:
        for i in range(4):
            curses.init_pair(i + 1, curses.COLOR_CYAN, -1)
        for i in range(4, 8):
            curses.init_pair(i + 1, curses.COLOR_MAGENTA if i < 7 else curses.COLOR_WHITE, -1)
        curses.init_pair(10, curses.COLOR_CYAN, -1)
        curses.init_pair(11, curses.COLOR_MAGENTA, -1)
        curses.init_pair(12, curses.COLOR_MAGENTA, -1)
        curses.init_pair(13, curses.COLOR_CYAN, -1)
        curses.init_pair(14, curses.COLOR_WHITE, -1)


def format_time(ms):
    s = max(0, int(ms / 1000))
    return f"{s // 60}:{s % 60:02d}"


def _safe(stdscr, y, x, text, attr=0):
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        pass



# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_bars(stdscr, bars, num_bands, top, bottom, left, right, bar_height, ce):
    """Draw frequency bars with galaxy gradient + star dissolution."""
    if bar_height < 2:
        return

    total = right - left
    bar_w = max(1, (total - (num_bands - 1)) // num_bands)
    gap = 1
    used = num_bands * bar_w + (num_bands - 1) * gap
    x_off = left + max(0, (total - used) // 2)

    for i in range(num_bands):
        bar_h = int(bars[i] * bar_height)
        if bar_h < 1 and bars[i] > 0.02:
            bar_h = 1
        x = x_off + i * (bar_w + gap)
        mid = bar_w // 2

        for row in range(bar_h):
            y = bottom - 1 - row
            if y < top or y >= bottom:
                continue

            if ce:
                abs_frac = row / max(bar_height - 1, 1)
                pair = min(int(abs_frac * 8), 7) + 1
            else:
                pair = 0

            bar_frac = row / max(bar_h - 1, 1)

            for dx in range(bar_w):
                cx = x + dx
                if cx >= right:
                    break

                if bar_h <= 2:
                    ch = "✦" if row == bar_h - 1 else "█"
                    if ch == "✦" and bar_w > 2 and dx != mid:
                        continue
                elif bar_h <= 5:
                    if bar_frac >= 0.85:
                        ch = "✦"
                        if bar_w > 2 and dx != mid:
                            continue
                    elif bar_frac >= 0.55:
                        ch = "▒"
                    elif bar_frac >= 0.25:
                        ch = "▓"
                    else:
                        ch = "█"
                else:
                    if bar_frac >= 0.90:
                        ch = "✦"
                        if bar_w > 2 and dx != mid:
                            continue
                    elif bar_frac >= 0.76:
                        ch = "░"
                        if bar_w > 3 and (dx == 0 or dx == bar_w - 1):
                            continue
                    elif bar_frac >= 0.58:
                        ch = "▒"
                    elif bar_frac >= 0.35:
                        ch = "▓"
                    else:
                        ch = "█"

                attr = curses.color_pair(pair) if ce else 0
                if bar_frac >= 0.76:
                    attr |= curses.A_BOLD
                _safe(stdscr, y, cx, ch, attr)


def _draw_separator(stdscr, y, w, ce):
    sep_width = min(w - 4, 56)
    dots = ""
    for j in range(sep_width):
        dots += "·" if j % 3 == 0 else " "
    x = max(1, (w - len(dots)) // 2)
    _safe(stdscr, y, x, dots[:w - x - 1],
          curses.color_pair(11) | curses.A_DIM if ce else curses.A_DIM)


def draw(stdscr, bars, spotify_poller, ce):
    h, w = stdscr.getmaxyx()
    stdscr.erase()

    num_bands = len(bars)
    if num_bands == 0 or w < 4 or h < 6:
        return

    has_track = spotify_poller and spotify_poller.track_name
    has_progress = spotify_poller and spotify_poller.duration_ms > 0
    footer_h = 2 if (has_track or has_progress) else 0

    bar_top = 0
    bar_bot = h - footer_h
    bar_height = bar_bot - bar_top

    # 1. Draw frequency bars across full width
    _draw_bars(stdscr, bars, num_bands, bar_top, bar_bot, 1, w - 1,
               bar_height, ce)

    # 2. Footer: separator + track info & progress
    if footer_h > 0:
        _draw_separator(stdscr, h - 2, w, ce)
        fy = h - 1

        # Track info (left-aligned)
        info_end = 1
        if has_track:
            _safe(stdscr, fy, 1, "✦",
                  curses.color_pair(12) | curses.A_BOLD if ce else curses.A_BOLD)
            info = f"{spotify_poller.track_name}  ·  {spotify_poller.artist}"
            max_info = w // 2 - 2
            if len(info) > max_info:
                info = info[:max_info - 3] + "..."
            _safe(stdscr, fy, 3, info,
                  curses.color_pair(10) if ce else curses.A_BOLD)
            info_end = 3 + len(info) + 2

        # Progress + time (right-aligned)
        if has_progress:
            play_icon = "⏸" if spotify_poller.is_playing else "▶"
            time_str = f"{play_icon} {format_time(spotify_poller.progress_ms)}/{format_time(spotify_poller.duration_ms)}"
            time_x = w - len(time_str) - 2
            _safe(stdscr, fy, time_x, time_str,
                  curses.color_pair(10) if ce else 0)

            # Progress bar fills gap between track info and time
            prog_x = info_end + 1
            prog_w = time_x - prog_x - 2
            if prog_w > 6:
                frac = spotify_poller.progress_ms / spotify_poller.duration_ms
                filled = int(frac * prog_w)
                for ci in range(prog_w):
                    cx = prog_x + ci
                    if ci == filled and filled < prog_w:
                        ch = "✦"
                        attr = curses.color_pair(12) | curses.A_BOLD if ce else curses.A_BOLD
                    elif ci < filled:
                        ch = "━"
                        attr = curses.color_pair(13) if ce else 0
                    else:
                        ch = "─"
                        attr = curses.color_pair(14) if ce else curses.A_DIM
                    _safe(stdscr, fy, cx, ch, attr)

    stdscr.refresh()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(33)

    ce = not args.color_off
    if ce:
        try:
            init_colors()
        except curses.error:
            ce = False

    spotify_poller = None
    sp = init_spotify()
    if sp:
        spotify_poller = SpotifyPoller(sp)
        spotify_poller.start()

    samplerate = 44100
    blocksize = 2048
    device = args.device_id

    if device is not None:
        try:
            info = sd.query_devices(device)
            samplerate = int(info["default_samplerate"])
        except Exception:
            pass

    capture = AudioCapture(device=device, samplerate=samplerate, blocksize=blocksize)
    capture.start()

    prev_bars = None
    running_peak = 1.0
    cached_art_url = ""

    try:
        while True:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                break

            h, w = stdscr.getmaxyx()
            num_bands = args.bars if args.bars else max(8, min(64, (w - 4) // 3))

            if prev_bars is None or len(prev_bars) != num_bands:
                prev_bars = np.zeros(num_bands)

            # Update bar colors from album art palette
            if spotify_poller and HAS_PIL and ce:
                art_url, art_data = spotify_poller.album_art
                if art_url != cached_art_url:
                    cached_art_url = art_url
                    palette = extract_palette(art_data, 8) if art_data else None
                    colors = palette if palette else GALAXY_GRADIENT
                    for i, c in enumerate(colors):
                        curses.init_pair(i + 1, c, -1)
                    # Theme UI text to match album palette
                    if palette:
                        curses.init_pair(10, palette[-2], -1)
                        curses.init_pair(11, palette[2], -1)
                        curses.init_pair(12, palette[-1], -1)
                        curses.init_pair(13, palette[-3], -1)
                    else:
                        curses.init_pair(10, 51, -1)
                        curses.init_pair(11, 99, -1)
                        curses.init_pair(12, 213, -1)
                        curses.init_pair(13, 51, -1)

            if spotify_poller and not spotify_poller.is_playing:
                prev_bars *= 0.8
                draw(stdscr, prev_bars, spotify_poller, ce)
                continue

            audio = capture.get_buffer()
            bands = compute_bands(audio, samplerate, num_bands)

            frame_peak = np.max(bands)
            if frame_peak > running_peak:
                running_peak = frame_peak
            else:
                running_peak *= 0.995

            running_peak = max(running_peak, 1e-6)
            bands_norm = np.clip(bands / running_peak, 0.0, 1.0)

            smoothed = np.where(
                bands_norm > prev_bars,
                prev_bars + (bands_norm - prev_bars) * 0.6,
                prev_bars * 0.7
            )

            prev_bars = smoothed
            draw(stdscr, smoothed, spotify_poller, ce)

    finally:
        capture.stop()
        if spotify_poller:
            spotify_poller.stop()


def run():
    parser = argparse.ArgumentParser(
        description="Spotify Terminal Visualizer — real-time frequency bars in your terminal"
    )
    parser.add_argument("--device", dest="device_flag", action="store_true",
                        help="Interactively choose an audio input device")
    parser.add_argument("--device-id", type=int, default=None, metavar="ID",
                        help="Use a specific audio device by ID")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--bars", type=int, default=0, metavar="N",
                        help="Fixed number of frequency bars (default: auto)")
    parser.add_argument("--color-off", action="store_true",
                        help="Disable colored output")
    args = parser.parse_args()

    if args.list_devices:
        devices = sd.query_devices()
        print("\nAvailable input devices:\n")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = " *" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']} ({d['max_input_channels']}ch, "
                      f"{int(d['default_samplerate'])}Hz){marker}")
        print()
        sys.exit(0)

    load_env()

    if args.device_flag and args.device_id is None:
        devices = sd.query_devices()
        print("\nAvailable input devices:\n")
        inputs = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = " *" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']} ({d['max_input_channels']}ch, "
                      f"{int(d['default_samplerate'])}Hz){marker}")
                inputs.append(i)
        print()
        try:
            choice = input("Select device number (Enter for default): ").strip()
            if choice:
                idx = int(choice)
                if idx in inputs:
                    args.device_id = idx
        except (ValueError, EOFError):
            pass

    print("Starting visualizer... (press q to quit)")
    if not HAS_PIL:
        print("(pip install Pillow for album art display)")

    sp = init_spotify()
    if sp:
        print("Spotify connected!")
    else:
        print("Spotify not connected — bars only, no track info.")

    time.sleep(0.5)
    curses.wrapper(lambda stdscr: main(stdscr, args))


if __name__ == "__main__":
    run()
