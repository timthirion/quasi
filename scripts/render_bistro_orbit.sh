#!/usr/bin/env bash
#
# scripts/render_bistro_orbit.sh — eight-frame orbit around the
# Bistro Exterior, look-at fixed, camera rotated around +Y in
# 45° steps. Frame 01 sits at the existing static hero's camera
# (data/output/bistro_reference.png); frame 04 is the user-
# requested 135°-rotated view.
#
# Camera math:
#   look_at        = (-25, 8, -20)
#   baseline pos   = (50, 6, 30)         # existing hero camera
#   baseline off   = pos - look_at = (75, -2, 50)
#   |off|_xz       ≈ 90.14
#
# Frame N (1-indexed) rotates the baseline offset by
# θ = (N-1) × 45° around +Y, then re-adds look_at:
#
#   01:   0°   → (50, 6, 30)
#   02:  45°   → (-7.32, 6, 68.39)
#   03:  90°   → (-75, 6, 55)
#   04: 135°   → (-113.39, 6, -2.32)   ← user-requested
#   05: 180°   → (-100, 6, -70)
#   06: 225°   → (-42.68, 6, -108.39)
#   07: 270°   → (25, 6, -95)
#   08: 315°   → (63.39, 6, -37.68)
#
# Sun + sun-color are world-fixed (directional light, not orbit-
# attached) so each frame sees the gothic façade lit from a
# different relative angle.
#
# Usage:
#   ./scripts/render_bistro_orbit.sh [frame_number] [spp]
#
#   No args         → renders all 8 frames at the default spp.
#   frame=N         → renders just frame N (1..8).
#   spp=N           → overrides default 1024 spp.
#
# Examples:
#   ./scripts/render_bistro_orbit.sh           # full orbit @ 1024 spp
#   ./scripts/render_bistro_orbit.sh 4 2048    # frame 04 only @ 2048 spp
#
# Output:
#   data/output/bistro_orbit_NN.{png,exr,_variance.png}
#
# Requires:
#   data/gltf/bistro/BistroExterior.gltf  (pulled via
#   scripts/fetch_bistro.py — see plan 0025 PT-bistro).

set -euo pipefail

FRAME="${1:-all}"
SPP="${2:-1024}"

SCENE="data/gltf/bistro/BistroExterior.gltf"
LOOK_AT="-25,8,-20"
FOV=50
SUN_DIR="0.2,1.0,-0.2"
SUN_COLOR="1.0,0.95,0.8"
SUN_INTENSITY=8
WIDTH=1024
HEIGHT=768

# Frame N → camera position triple. Strings are space-separated
# x y z, formatted to two decimals.
declare -a CAMERA_POS=(
  ""                       # 0 — unused (frames are 1-indexed)
  "50,6,30"                # 01:   0°
  "-7.32,6,68.39"          # 02:  45°
  "-75,6,55"               # 03:  90°
  "-113.39,6,-2.32"        # 04: 135°  ← user-requested
  "-100,6,-70"             # 05: 180°
  "-42.68,6,-108.39"       # 06: 225°
  "25,6,-95"               # 07: 270°
  "63.39,6,-37.68"         # 08: 315°
)

render_frame() {
  local n="$1"
  local pos="${CAMERA_POS[$n]}"
  local out
  out=$(printf "data/output/bistro_orbit_%02d" "$n")
  echo "[orbit] frame $n: --camera-pos $pos --spp $SPP → $out.png"
  cargo run --release -- render \
    --scene "$SCENE" \
    --camera-pos "$pos" \
    --look-at "$LOOK_AT" \
    --fov "$FOV" \
    --sun-dir "$SUN_DIR" \
    --sun-color "$SUN_COLOR" \
    --sun-intensity "$SUN_INTENSITY" \
    --width "$WIDTH" --height "$HEIGHT" --spp "$SPP" \
    --out "$out"
}

if [[ "$FRAME" == "all" ]]; then
  for n in 1 2 3 4 5 6 7 8; do
    render_frame "$n"
  done
else
  render_frame "$FRAME"
fi
