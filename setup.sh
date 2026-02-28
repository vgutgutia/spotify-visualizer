#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Spotify Terminal Visualizer Setup    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is required but not found. Install it first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python 3 found: $(python3 --version)"

# Install Python dependencies
echo
echo -e "${CYAN}Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Check for BlackHole (optional, macOS only)
echo
if [[ "$OSTYPE" == "darwin"* ]]; then
    if system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole"; then
        echo -e "${GREEN}✓${NC} BlackHole virtual audio device detected"
        echo "  You can route system audio through BlackHole for best results."
        echo "  Set up a Multi-Output Device in Audio MIDI Setup to hear audio + capture it."
    else
        echo -e "${YELLOW}Optional:${NC} BlackHole virtual audio device not found."
        echo "  Without it, the visualizer will use your microphone input."
        echo "  For system audio capture, install BlackHole:"
        echo -e "  ${CYAN}brew install blackhole-2ch${NC}"
        echo "  Then create a Multi-Output Device in Audio MIDI Setup"
        echo "  (combining your speakers + BlackHole) so you hear audio while capturing it."
    fi
fi

# Spotify setup
echo
echo -e "${CYAN}━━━ Spotify Integration (optional) ━━━${NC}"
echo
echo "The visualizer works without Spotify — it just won't show track info."
echo "To enable Spotify integration, you need a Spotify Developer app."
echo

read -p "Set up Spotify integration now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    echo -e "${YELLOW}Steps to create a Spotify app:${NC}"
    echo "  1. Go to https://developer.spotify.com/dashboard"
    echo "  2. Click 'Create App'"
    echo "  3. Name: anything (e.g. 'Terminal Visualizer')"
    echo "  4. Redirect URI: https://127.0.0.1:8888/callback"
    echo "  5. Check 'Web API'"
    echo "  6. Click 'Save'"
    echo "  7. Go to Settings and copy your Client ID and Client Secret"
    echo

    read -p "Enter your Spotify Client ID: " client_id
    read -p "Enter your Spotify Client Secret: " client_secret

    if [[ -n "$client_id" && -n "$client_secret" ]]; then
        cat > .env <<EOF
SPOTIPY_CLIENT_ID=${client_id}
SPOTIPY_CLIENT_SECRET=${client_secret}
SPOTIPY_REDIRECT_URI=https://127.0.0.1:8888/callback
EOF
        echo
        echo -e "${GREEN}✓${NC} Credentials saved to .env"
        echo "  The visualizer will read these automatically."
    else
        echo -e "${YELLOW}Skipped — no credentials entered.${NC}"
    fi
else
    echo "Skipped. You can run the visualizer with --no-spotify or set up later."
fi

echo
echo -e "${GREEN}━━━ Setup complete! ━━━${NC}"
echo
echo "Usage:"
echo -e "  ${CYAN}python3 visualizer.py${NC}              # with Spotify + mic input"
echo -e "  ${CYAN}python3 visualizer.py --no-spotify${NC}  # without Spotify"
echo -e "  ${CYAN}python3 visualizer.py --device${NC}      # choose audio device"
echo -e "  ${CYAN}python3 visualizer.py --help${NC}        # all options"
echo
