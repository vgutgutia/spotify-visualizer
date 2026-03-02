#!/usr/bin/env python3
"""Spotify Terminal Visualizer — real-time frequency bars from mic input + Spotify metadata."""

import argparse
import curses
import io
import os
import random
import sys
import threading
import time
import urllib.request
from collections import deque

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
GALAXY_UI_COLORS = [51, 99, 213, 51]  # pairs 10, 11, 12, 13
PARTICLE_CHARS = ["✦", "·", "*", "+"]
MODE_NAMES = ["BARS", "MIRROR", "WAVE", "SPECTROGRAM"]


def _rgb_to_256(r, g, b):
    """Map RGB to xterm-256 color index."""
    ri = round(r / 255 * 5)
    gi = round(g / 255 * 5)
    bi = round(b / 255 * 5)
    return 16 + 36 * ri + 6 * gi + bi


def _256_to_rgb(idx):
    """Reverse lookup from xterm-256 index to (r, g, b)."""
    if 16 <= idx <= 231:
        idx -= 16
        r = idx // 36
        g = (idx // 6) % 6
        b = idx % 6
        return (r * 51, g * 51, b * 51)
    elif 232 <= idx <= 255:
        v = 8 + (idx - 232) * 10
        return (v, v, v)
    basic = [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
        (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ]
    if 0 <= idx < 16:
        return basic[idx]
    return (128, 128, 128)


GALAXY_FULL_RGB = [_256_to_rgb(c) for c in GALAXY_GRADIENT] + \
                  [_256_to_rgb(c) for c in GALAXY_UI_COLORS]


def _lerp_palette(old_rgb, tgt_rgb, t):
    """Interpolate between two RGB palette lists and apply to curses color pairs."""
    result = []
    for (cr, cg, cb), (tr, tg, tb) in zip(old_rgb, tgt_rgb):
        r = int(cr + (tr - cr) * t)
        g = int(cg + (tg - cg) * t)
        b = int(cb + (tb - cb) * t)
        result.append((r, g, b))
    for i in range(min(8, len(result))):
        curses.init_pair(i + 1, _rgb_to_256(*result[i]), -1)
    ui_pairs = [10, 11, 12, 13]
    for i in range(min(4, len(result) - 8)):
        curses.init_pair(ui_pairs[i], _rgb_to_256(*result[8 + i]), -1)
    return result


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

def _draw_bars(stdscr, bars, num_bands, top, bottom, left, right,
               bar_height, ce, beat_flash=0, peak_heights=None):
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
                if beat_flash > 0 or bar_frac >= 0.76:
                    attr |= curses.A_BOLD
                _safe(stdscr, y, cx, ch, attr)

    # Peak hold dots
    if peak_heights is not None and ce:
        mid = bar_w // 2
        for i in range(num_bands):
            peak_h = int(peak_heights[i] * bar_height)
            if peak_h > 0:
                y = bottom - 1 - peak_h
                if top <= y < bottom:
                    px = x_off + i * (bar_w + gap) + mid
                    if px < right:
                        _safe(stdscr, y, px, "·",
                              curses.color_pair(8) | curses.A_BOLD)


def _draw_mirror(stdscr, bars, num_bands, top, bottom, left, right,
                 bar_height, ce, beat_flash=0, peak_heights=None):
    """Draw mirrored bars growing from horizontal center up and down."""
    if bar_height < 4:
        return

    total = right - left
    bar_w = max(1, (total - (num_bands - 1)) // num_bands)
    gap = 1
    used = num_bands * bar_w + (num_bands - 1) * gap
    x_off = left + max(0, (total - used) // 2)
    mid_w = bar_w // 2

    center_y = (top + bottom) // 2
    half_height = max((bottom - top) // 2, 1)

    for i in range(num_bands):
        bar_h = int(bars[i] * half_height)
        if bar_h < 1 and bars[i] > 0.02:
            bar_h = 1
        x = x_off + i * (bar_w + gap)

        for row in range(bar_h):
            y_up = center_y - 1 - row
            y_down = center_y + row

            if ce:
                abs_frac = row / max(half_height - 1, 1)
                pair = min(int(abs_frac * 8), 7) + 1
            else:
                pair = 0

            bar_frac = row / max(bar_h - 1, 1)

            if bar_frac >= 0.90:
                ch = "✦"
            elif bar_frac >= 0.76:
                ch = "░"
            elif bar_frac >= 0.58:
                ch = "▒"
            elif bar_frac >= 0.35:
                ch = "▓"
            else:
                ch = "█"

            attr = curses.color_pair(pair) if ce else 0
            if beat_flash > 0 or bar_frac >= 0.76:
                attr |= curses.A_BOLD

            for dx in range(bar_w):
                cx = x + dx
                if cx >= right:
                    break
                if ch == "✦" and bar_w > 2 and dx != mid_w:
                    continue
                if ch == "░" and bar_w > 3 and (dx == 0 or dx == bar_w - 1):
                    continue
                if top <= y_up < bottom:
                    _safe(stdscr, y_up, cx, ch, attr)
                if top <= y_down < bottom:
                    _safe(stdscr, y_down, cx, ch, attr)

    # Peak hold dots (mirrored)
    if peak_heights is not None and ce:
        for i in range(num_bands):
            peak_h = int(peak_heights[i] * half_height)
            if peak_h > 0:
                y_up = center_y - 1 - peak_h
                y_down = center_y + peak_h
                px = x_off + i * (bar_w + gap) + mid_w
                if px < right:
                    peak_attr = curses.color_pair(8) | curses.A_BOLD
                    if top <= y_up < bottom:
                        _safe(stdscr, y_up, px, "·", peak_attr)
                    if top <= y_down < bottom:
                        _safe(stdscr, y_down, px, "·", peak_attr)


def _draw_wave(stdscr, audio_buf, top, bottom, left, right, ce, beat_flash=0):
    """Draw audio waveform using line-drawing characters."""
    h = bottom - top
    w = right - left
    if h < 3 or w < 3 or audio_buf is None:
        return

    center = top + h // 2
    buf_len = len(audio_buf)
    step = max(1, buf_len // w)

    prev_y = None
    for col in range(w):
        idx = min(col * step, buf_len - 1)
        val = audio_buf[idx]
        amp = min(abs(val) * 3, 1.0)

        y = center - int(val * (h // 2) * 0.8)
        y = max(top, min(bottom - 1, y))

        if ce:
            pair = min(int(amp * 8), 7) + 1
        else:
            pair = 0

        attr = curses.color_pair(pair) if ce else 0
        if beat_flash > 0 or amp > 0.7:
            attr |= curses.A_BOLD

        x = left + col

        if prev_y is not None:
            dy = y - prev_y
            if dy == 0:
                ch = "─"
            elif dy < -1:
                ch = "╱"
            elif dy > 1:
                ch = "╲"
            elif dy == -1:
                ch = "╱"
            else:
                ch = "╲"
            # Fill vertical gaps between consecutive points
            if abs(y - prev_y) > 1:
                step_y = 1 if y > prev_y else -1
                for fy in range(prev_y + step_y, y, step_y):
                    if top <= fy < bottom:
                        _safe(stdscr, fy, x, "│", attr)
        else:
            ch = "─"

        if top <= y < bottom:
            _safe(stdscr, y, x, ch, attr)

        prev_y = y


def _draw_spectrogram(stdscr, spec_history, num_bands, top, bottom,
                      left, right, ce, beat_flash=0):
    """Draw scrolling spectrogram heat map."""
    h = bottom - top
    w = right - left
    if h < 3 or w < 3 or not spec_history:
        return

    SPEC_CHARS = " ·░▒▓█"
    n_rows = min(num_bands, h)

    for col_idx, col_data in enumerate(spec_history):
        x = left + col_idx
        if x >= right:
            break

        n_data = len(col_data)
        for row in range(n_rows):
            y = bottom - 1 - row
            if y < top:
                break

            band_idx = min(int(row * n_data / n_rows), n_data - 1)
            val = min(col_data[band_idx], 1.0)

            ci = min(int(val * (len(SPEC_CHARS) - 1)), len(SPEC_CHARS) - 1)
            ch = SPEC_CHARS[ci]

            if ch == " ":
                continue

            if ce:
                pair = min(int(val * 8), 7) + 1
            else:
                pair = 0

            attr = curses.color_pair(pair) if ce else 0
            if beat_flash > 0:
                attr |= curses.A_BOLD
            _safe(stdscr, y, x, ch, attr)


def _draw_particles(stdscr, particles, top, bottom, left, right, ce):
    """Draw active particles as overlay."""
    for p in particles:
        x = int(round(p["x"]))
        y = int(round(p["y"]))
        if x < left or x >= right or y < top or y >= bottom:
            continue
        if p["life"] <= 0:
            continue
        attr = curses.color_pair(p["color_pair"]) if ce else 0
        if p["life"] < p["max_life"] * 0.3:
            attr |= curses.A_DIM
        else:
            attr |= curses.A_BOLD
        _safe(stdscr, y, x, p["char"], attr)


def _draw_separator(stdscr, y, w, ce):
    sep_width = min(w - 4, 56)
    dots = ""
    for j in range(sep_width):
        dots += "·" if j % 3 == 0 else " "
    x = max(1, (w - len(dots)) // 2)
    _safe(stdscr, y, x, dots[:w - x - 1],
          curses.color_pair(11) | curses.A_DIM if ce else curses.A_DIM)


def draw(stdscr, bars, spotify_poller, ce, no_footer=False,
         viz_mode=0, beat_flash=0, peak_heights=None,
         particles=None, spec_history=None, mode_label_timer=0,
         audio_buf=None):
    h, w = stdscr.getmaxyx()
    stdscr.erase()

    num_bands = len(bars)
    if num_bands == 0 or w < 4 or h < 6:
        return

    has_track = not no_footer and spotify_poller and spotify_poller.track_name
    has_progress = not no_footer and spotify_poller and spotify_poller.duration_ms > 0
    footer_h = 2 if (has_track or has_progress) else 0

    bar_top = 0
    bar_bot = h - footer_h
    bar_height = bar_bot - bar_top

    # Dispatch visualization mode
    if viz_mode == 1:
        _draw_mirror(stdscr, bars, num_bands, bar_top, bar_bot, 1, w - 1,
                     bar_height, ce, beat_flash, peak_heights)
    elif viz_mode == 2:
        _draw_wave(stdscr, audio_buf, bar_top, bar_bot, 1, w - 1,
                   ce, beat_flash)
    elif viz_mode == 3:
        _draw_spectrogram(stdscr, spec_history, num_bands, bar_top, bar_bot,
                          1, w - 1, ce, beat_flash)
    else:
        _draw_bars(stdscr, bars, num_bands, bar_top, bar_bot, 1, w - 1,
                   bar_height, ce, beat_flash, peak_heights)

    # Particles overlay (skip in spectrogram mode)
    if viz_mode != 3 and particles:
        _draw_particles(stdscr, particles, bar_top, bar_bot, 1, w - 1, ce)

    # Mode label overlay
    if mode_label_timer > 0 and 0 <= viz_mode < len(MODE_NAMES):
        label = f"✦ {MODE_NAMES[viz_mode]}"
        lx = max(1, (w - len(label)) // 2)
        ly = max(0, bar_top + 1)
        if ly < bar_bot:
            _safe(stdscr, ly, lx, label,
                  curses.color_pair(12) | curses.A_BOLD if ce else curses.A_BOLD)

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

    # Visual feature state
    viz_mode = 0
    mode_label_timer = 0
    beat_flash = 0
    bass_avg = 0.0
    peak_heights = None
    peak_timers = None
    particles = []
    spec_history = deque(maxlen=200)
    current_rgb = list(GALAXY_FULL_RGB)
    old_rgb = list(GALAXY_FULL_RGB)
    target_rgb = list(GALAXY_FULL_RGB)
    palette_progress = 1.0

    try:
        while True:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("m"), ord("M")):
                viz_mode = (viz_mode + 1) % 4
                mode_label_timer = 20

            h, w = stdscr.getmaxyx()
            num_bands = args.bars if args.bars else max(8, min(64, (w - 4) // 3))

            if prev_bars is None or len(prev_bars) != num_bands:
                prev_bars = np.zeros(num_bands)
                peak_heights = np.zeros(num_bands)
                peak_timers = np.zeros(num_bands)

            # Resize spectrogram history deque on terminal width change
            desired_maxlen = max(10, w - 2)
            if spec_history.maxlen != desired_maxlen:
                spec_history = deque(spec_history, maxlen=desired_maxlen)

            # Smooth palette transitions
            if spotify_poller and HAS_PIL and ce:
                art_url, art_data = spotify_poller.album_art
                if art_url != cached_art_url:
                    cached_art_url = art_url
                    palette = extract_palette(art_data, 8) if art_data else None
                    if palette:
                        new_target = [_256_to_rgb(c) for c in palette]
                        new_target += [_256_to_rgb(palette[-2]),
                                       _256_to_rgb(palette[2]),
                                       _256_to_rgb(palette[-1]),
                                       _256_to_rgb(palette[-3])]
                    else:
                        new_target = list(GALAXY_FULL_RGB)
                    old_rgb = list(current_rgb)
                    target_rgb = new_target
                    palette_progress = 0.0

            if palette_progress < 1.0 and ce:
                palette_progress = min(palette_progress + 0.03, 1.0)
                current_rgb = _lerp_palette(old_rgb, target_rgb, palette_progress)

            # Paused: decay bars/peaks, update particles, draw, continue
            if spotify_poller and not spotify_poller.is_playing:
                prev_bars *= 0.8
                if peak_heights is not None:
                    peak_heights *= 0.92
                # Let particles fade out
                new_particles = []
                for p in particles:
                    p["y"] -= p["vy"]
                    p["x"] += p["vx"]
                    p["life"] -= 1
                    if p["life"] > 0:
                        new_particles.append(p)
                particles = new_particles
                draw(stdscr, prev_bars, spotify_poller, ce, args.no_footer,
                     viz_mode=viz_mode, beat_flash=beat_flash,
                     peak_heights=peak_heights, particles=particles,
                     spec_history=spec_history,
                     mode_label_timer=mode_label_timer)
                if beat_flash > 0:
                    beat_flash -= 1
                if mode_label_timer > 0:
                    mode_label_timer -= 1
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

            # Beat detection — bass energy from first 3 bands
            bass_energy = float(np.mean(smoothed[:min(3, num_bands)]))
            bass_avg = bass_avg * 0.9 + bass_energy * 0.1
            beat_triggered = bass_energy > bass_avg * 1.6 and bass_energy > 0.05
            if beat_triggered:
                beat_flash = 4

            # Peak hold update
            for i in range(num_bands):
                if smoothed[i] > peak_heights[i]:
                    peak_heights[i] = smoothed[i]
                    peak_timers[i] = 20
                else:
                    peak_timers[i] -= 1
                    if peak_timers[i] <= 0:
                        peak_heights[i] *= 0.92

            # Append to spectrogram history
            if viz_mode == 3:
                spec_history.append(smoothed.copy())

            # Particle spawn & physics
            has_track_info = (not args.no_footer and spotify_poller
                              and spotify_poller.track_name)
            has_progress_info = (not args.no_footer and spotify_poller
                                 and spotify_poller.duration_ms > 0)
            footer_h = 2 if (has_track_info or has_progress_info) else 0
            bar_bot = h - footer_h
            bar_height_calc = bar_bot  # bar_top is 0

            if viz_mode != 3 and bar_height_calc > 2:
                total = (w - 1) - 1
                bar_w_calc = max(1, (total - (num_bands - 1)) // num_bands)
                gap_calc = 1
                used_calc = num_bands * bar_w_calc + (num_bands - 1) * gap_calc
                x_off_calc = 1 + max(0, (total - used_calc) // 2)

                # Spawn particles from bars above 60%
                for i in range(num_bands):
                    if (smoothed[i] > 0.6 and random.random() < 0.15
                            and len(particles) < 80):
                        bar_h = int(smoothed[i] * bar_height_calc)
                        px = (x_off_calc + i * (bar_w_calc + gap_calc)
                              + bar_w_calc // 2)
                        py = bar_bot - 1 - bar_h
                        pair = min(int(smoothed[i] * 8), 7) + 1 if ce else 1
                        particles.append({
                            "x": float(px), "y": float(py),
                            "vx": random.uniform(-0.2, 0.2),
                            "vy": random.uniform(0.3, 0.8),
                            "char": random.choice(PARTICLE_CHARS),
                            "life": random.randint(15, 30),
                            "max_life": 30,
                            "color_pair": pair,
                        })

                # Beat-triggered extra particles from loudest bars
                if beat_triggered:
                    loudest = np.argsort(smoothed)[-3:]
                    for idx in loudest:
                        for _ in range(random.randint(1, 2)):
                            if len(particles) >= 80:
                                break
                            bar_h = int(smoothed[idx] * bar_height_calc)
                            px = (x_off_calc + idx * (bar_w_calc + gap_calc)
                                  + bar_w_calc // 2)
                            py = bar_bot - 1 - bar_h
                            pair = (min(int(smoothed[idx] * 8), 7) + 1
                                    if ce else 1)
                            particles.append({
                                "x": float(px), "y": float(py),
                                "vx": random.uniform(-0.3, 0.3),
                                "vy": random.uniform(0.5, 1.0),
                                "char": random.choice(PARTICLE_CHARS),
                                "life": random.randint(20, 35),
                                "max_life": 35,
                                "color_pair": pair,
                            })

            # Update particle physics
            new_particles = []
            for p in particles:
                p["y"] -= p["vy"]
                p["x"] += p["vx"]
                p["life"] -= 1
                if p["life"] > 0:
                    new_particles.append(p)
            particles = new_particles

            # Draw current frame
            draw(stdscr, smoothed, spotify_poller, ce, args.no_footer,
                 viz_mode=viz_mode, beat_flash=beat_flash,
                 peak_heights=peak_heights, particles=particles,
                 spec_history=spec_history,
                 mode_label_timer=mode_label_timer,
                 audio_buf=audio)

            # Decrement timers after draw
            if beat_flash > 0:
                beat_flash -= 1
            if mode_label_timer > 0:
                mode_label_timer -= 1

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
    parser.add_argument("--no-footer", action="store_true",
                        help="Hide track info and progress bar footer")
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

    print("Starting visualizer... (press q to quit, m to change mode)")
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
