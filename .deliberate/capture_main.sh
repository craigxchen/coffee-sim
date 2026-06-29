#!/bin/zsh
# Capture the OLD MPM solver (main branch) renders for the 3-solver comparison.
# main's web app already built at: coffee-main-wt/crates/sim-wasm/www-3d (pkg/ present).
# Canvas id is #sim-canvas (same as rewrite). Debug scenes are [data-debug-scene="..."] buttons
# under the #scene-tab-debug panel; Center Pour is #scene-center-pour under #scene-tab-main.
set -u
WWW=/Users/cxc/Github/coffee-main-wt/crates/sim-wasm/www-3d
OUT=/Users/cxc/Github/coffee-sim/.deliberate/renders/main
PROG=/tmp/maincap.progress
: > "$PROG"
mkdir -p "$OUT"

echo "[serve] starting http.server :8001" >>"$PROG"
python3 -m http.server 8001 --directory "$WWW" >/tmp/main8001.log 2>&1 &
SRV=$!
# wait for server
for i in {1..30}; do curl -s -o /dev/null http://localhost:8001/ && break; sleep 0.3; done

AB() { agent-browser --session maincap "$@" >>/tmp/maincap.log 2>&1; }
SHOT() { agent-browser --session maincap screenshot '#sim-canvas' "$1" >>/tmp/maincap.log 2>&1; echo "[shot] $1" >>"$PROG"; }

echo "[open] http://localhost:8001" >>"$PROG"
AB open http://localhost:8001
AB wait 5000
agent-browser --session maincap eval 'navigator.gpu ? "gpu-ok" : "no-gpu"' >"$OUT/diag.txt" 2>&1
echo "[diag] $(cat "$OUT/diag.txt")" >>"$PROG"

cap_debug() { # $1=debug-scene-id  $2=colname  $3 $4 $5 = wait ms before t0/t1/t2
  echo "[scene] $2 ($1)" >>"$PROG"
  AB click '#scene-tab-debug'; AB wait 400
  AB click "[data-debug-scene=\"$1\"]"
  AB wait "$3"; SHOT "$OUT/$2__main__t0.png"
  AB wait "$4"; SHOT "$OUT/$2__main__t1.png"
  AB wait "$5"; SHOT "$OUT/$2__main__t2.png"
}

cap_debug dam-break-slosh   box_slosh     2500 5000 8000
cap_debug sparse-free-jet   water_only    2500 5000 8000
cap_debug filter-water-block sand_wall_box 2500 6000 10000

# cup_drip_on_pool: center-pour run LONG so the cup fills and slow drips reveal stir/blow-up
echo "[scene] cup_drip_on_pool (center-pour, long)" >>"$PROG"
AB click '#scene-tab-main'; AB wait 400
AB click '#scene-center-pour'
AB wait 4000;  SHOT "$OUT/cup_drip_on_pool__main__t0.png"
AB wait 12000; SHOT "$OUT/cup_drip_on_pool__main__t1.png"
AB wait 16000; SHOT "$OUT/cup_drip_on_pool__main__t2.png"

agent-browser --session maincap close >>/tmp/maincap.log 2>&1
kill $SRV 2>/dev/null
echo "[done] $(ls -1 "$OUT"/*.png 2>/dev/null | wc -l) png(s)" >>"$PROG"
echo DONE
