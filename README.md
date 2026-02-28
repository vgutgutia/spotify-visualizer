<p align="center">
  <h1 align="center">♫ spotify-visualizer</h1>
  <p align="center">
    <strong>Real-time audio visualizer for your terminal, synced with Spotify</strong>
  </p>
  <p align="center">
    Frequency bars that dance to your music, colored by your album art.<br>
    Connect Spotify and the entire theme shifts to match whatever you're listening to.
  </p>
  <p align="center">
    <a href="#install"><img src="https://img.shields.io/badge/install-setup.sh-blue?style=flat-square" alt="Install"></a>
    <img src="https://img.shields.io/badge/python-3.6%2B-yellow?style=flat-square&logo=python&logoColor=white" alt="Python 3.6+">
    <img src="https://img.shields.io/badge/spotify-integrated-1DB954?style=flat-square&logo=spotify&logoColor=white" alt="Spotify">
    <img src="https://img.shields.io/badge/license-MIT-gray?style=flat-square" alt="MIT License">
  </p>
</p>

<br>

<p align="center">

```
      ✦           ✦                   ✦
      █ ✦       ✦ █               ✦   █
      █ ▓     █ ▒ █   ✦       ✦  █ ✦ █ ✦
    █ █ █ ✦ █ █ ░ █   █   ✦ █ ▓ █ █ ▒ █ █
    █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █
  · · · · · · · · · · · · · · · · · · · · · · ·
  ✦ Blinding Lights  ·  The Weeknd  ━━━━━━━✦────  ⏸ 2:14/3:20
```

</p>

<br>

## Features

- **Album-reactive colors** — extracts dominant colors from the current album art and themes the entire visualizer to match
- **FFT frequency bars** — real-time Fast Fourier Transform splits audio into log-spaced frequency bands
- **Smooth animations** — bars rise fast and decay gradually with star dissolution at the peaks (`█ → ▓ → ▒ → ░ → ✦`)
- **Dynamic range** — running peak normalization so quiet tracks look different from loud ones
- **Spotify metadata** — track name, artist, and a live progress bar in the footer
- **Galaxy fallback** — when Spotify isn't connected, bars use a blue → cyan → purple → pink → white gradient
- **Adaptive layout** — bar count and sizing scale automatically with terminal width
- **Zero config start** — works immediately with your microphone; optionally capture system audio with BlackHole

## Install

```sh
git clone https://github.com/vgutgutia/spotify-visualizer.git
cd spotify-visualizer
bash setup.sh
```

The setup script installs dependencies, checks for [BlackHole](https://github.com/ExistentialAudio/BlackHole) (optional, for system audio capture), and walks you through Spotify app creation.

**Manual install:**

```sh
git clone https://github.com/vgutgutia/spotify-visualizer.git
cd spotify-visualizer
pip install -r requirements.txt
```

## Usage

```sh
# Start with Spotify integration + mic input
python3 visualizer.py

# Choose a specific audio input device
python3 visualizer.py --device

# Use a specific device by ID
python3 visualizer.py --device-id 2

# List available audio devices
python3 visualizer.py --list-devices

# Fixed number of bars
python3 visualizer.py --bars 24

# Disable color (for terminals without 256-color support)
python3 visualizer.py --color-off
```

Press **q** or **Esc** to quit.

## Spotify Setup

The visualizer works without Spotify — you just won't see track info or album-reactive colors. To enable it:

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Set the redirect URI to `https://127.0.0.1:8888/callback`
4. Check **Web API** and save
5. Copy your **Client ID** and **Client Secret**
6. Create a `.env` file in the project directory:

```sh
SPOTIPY_CLIENT_ID=your_client_id_here
SPOTIPY_CLIENT_SECRET=your_client_secret_here
SPOTIPY_REDIRECT_URI=https://127.0.0.1:8888/callback
```

On first run, a browser window will open for Spotify authorization. After that, the token is cached.

## How It Works

1. **Audio capture** — `sounddevice.InputStream` reads audio samples from the selected input device in a background thread
2. **FFT processing** — applies a Hanning window, runs `numpy.fft.rfft`, then groups magnitudes into log-spaced frequency bands (60 Hz – 16 kHz)
3. **Normalization** — a running peak tracker (instant rise, 0.995 decay per frame) preserves dynamic range between quiet and loud passages
4. **Smooth animation** — bars interpolate toward their target (fast attack 0.6, gradual release 0.7) for fluid motion
5. **Palette extraction** — when the track changes, PIL quantizes the album art into dominant colors, boosts saturation, and maps them to the bar gradient and UI elements
6. **curses rendering** — draws at ~30 FPS using block characters with star dissolution at the peaks

## Audio Source

By default, the visualizer listens to your **microphone**. For best results, capture system audio directly using [BlackHole](https://github.com/ExistentialAudio/BlackHole):

```sh
brew install blackhole-2ch
```

Then in **Audio MIDI Setup** (macOS):
1. Create a **Multi-Output Device** combining your speakers + BlackHole
2. Set it as your system output (so you still hear audio)
3. Run the visualizer with `--device` and select BlackHole as the input

## Requirements

- macOS (or Linux with PulseAudio/ALSA)
- Python 3.6+
- Terminal with 256-color support (most modern terminals)

## License

MIT — [Vansh Gutgutia](https://github.com/vgutgutia)
