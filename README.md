# gestureverse_anandaswaroop
# Gesture-Controlled Subway Surfers (Linux)

Control **Subway Surfers** on Linux using **hand gestures**, a webcam, and computer vision.  
This project maps real-time hand gestures to **keyboard arrow key events**, allowing you to play the game without touching the keyboard.

Built for **Ubuntu (X11)** and tested with **AppImage-based games**.

---

## Features

- Real-time **hand tracking** using MediaPipe
- Gesture → **Arrow key tap mapping** (game-safe input)
- Works with **AppImage games** (native Linux apps)
- Human-like key injection using `xdotool`
- Adjustable cooldowns for smooth, non-spammy control
- Designed specifically for **gesture-based gaming**, not text input

---

## Gesture Mapping

| Gesture | Action | Key |
|------|------|----|
| Pinch (thumb + index) | Jump | `Up` |
| Left hand fist | Move left | `Left` |
| Right hand fist | Move right | `Right` |
| Right index finger only | Roll / Slide | `Down` |

Gestures are **tap-based**, not hold-based — this is critical for games.

---

## Tech Stack

- **Python 3**
- **OpenCV** – camera input & visualization
- **MediaPipe Tasks API** – hand landmark detection
- **xdotool** – low-level X11 keyboard event injection
- **Linux (Xorg)** – required for global input control

---

## System Requirements (Important)

This project **will NOT work on Wayland**.

You must be running:
- **Ubuntu on Xorg (X11)**

Check your session:
```bash
echo $XDG_SESSION_TYPE
