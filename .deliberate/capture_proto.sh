#!/bin/zsh
# Build the PROTOTYPE twofield (state-dependent ballistic-preserve in g2p_water) and capture it
# on the 4 comparison scenes, into .deliberate/renders/proto/, to compare vs the slime baseline.
set -u
cd /Users/cxc/Github/coffee-sim
OUT=/Users/cxc/Github/coffee-sim/.deliberate/renders/proto
PROG=/tmp/protocap.progress
: > "$PROG"; mkdir -p "$OUT"

echo "[build] wasm-pack release" >>"$PROG"
wasm-pack build --target web --out-dir www/pkg --release >/tmp/proto_build.log 2>&1
echo "[build] exit=$? (tail: $(tail -1 /tmp/proto_build.log))" >>"$PROG"

pkill -f "http.server 8000" 2>/dev/null; sleep 0.5
python3 -m http.server 8000 --directory www >/tmp/proto8000.log 2>&1 &
SRV=$!
for i in {1..40}; do curl -s -o /dev/null http://localhost:8000/ && break; sleep 0.3; done

agent-browser --session proto close 2>/dev/null
AB() { agent-browser --session proto "$@" >>/tmp/protocap.log 2>&1; }
SHOT() { agent-browser --session proto screenshot '#sim-canvas' "$1" >>/tmp/protocap.log 2>&1; echo "[shot] $1" >>"$PROG"; }

AB open http://localhost:8000
AB wait 5000
agent-browser --session proto eval 'navigator.gpu ? "gpu-ok" : "no-gpu"' >"$OUT/diag.txt" 2>&1
# robustly select the twofield solver and fire change so the app rebuilds with it
agent-browser --session proto eval '(()=>{const s=document.querySelector("#solver-select");const o=[...s.options].find(o=>/two.?field/i.test(o.textContent)||/two.?field/i.test(o.value));if(o){s.value=o.value;s.dispatchEvent(new Event("change",{bubbles:true}));return "selected:"+o.value}return "OPTS:"+[...s.options].map(o=>o.value+"="+o.textContent).join(",")})()' >"$OUT/solver.txt" 2>&1
echo "[solver] $(cat "$OUT/solver.txt")" >>"$PROG"
AB wait 1500

cap() { # $1 button  $2 col  $3 $4 $5 waits(ms)
  echo "[scene] $2" >>"$PROG"
  AB click "$1"
  AB wait "$3"; SHOT "$OUT/$2__twofield__t0.png"
  AB wait "$4"; SHOT "$OUT/$2__twofield__t1.png"
  AB wait "$5"; SHOT "$OUT/$2__twofield__t2.png"
}
cap '#scene-box-slosh'        BoxSlosh      2500 5000 8000
cap '#scene-free-stream'      WaterOnly     2500 5000 8000
cap '#scene-cup-drip-on-pool' CupDripOnPool 4000 12000 16000
cap '#scene-sand-wall-box'    SandWallBox   2500 6000 10000

agent-browser --session proto close >>/tmp/protocap.log 2>&1
kill $SRV 2>/dev/null
echo "[done] $(ls -1 "$OUT"/*.png 2>/dev/null | wc -l) png(s)" >>"$PROG"
echo DONE
