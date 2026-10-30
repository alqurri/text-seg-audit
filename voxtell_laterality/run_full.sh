#!/bin/bash
# Full laterality audit. Usage: VX=/path/to/voxtell/python TS=/path/to/totalseg/python DATA=/path/to/data W=$DATA/voxtell_work bash run_full.sh
set -eo pipefail
: "${VX:=python}" "${TS:=python}" "${DATA:?set DATA}" "${W:=$DATA/voxtell_work}"
cd "$(dirname "$0")"
mkdir -p "$W"
echo "== download $(date)";  $VX download.py --rex-dir $DATA/rexgroundingct --ctrate-dir $DATA/ctrate
echo "== prepare $(date)";   $VX prepare.py --rex-dir $DATA/rexgroundingct --ctrate-dir $DATA/ctrate --work $W 2>&1 | tee $W/prepare.log
echo "== lungs $(date)";     $TS -u lungs.py --work $W --fast
echo "== infer $(date)";     $VX -u infer_voxtell.py --work $W
echo "== analyze $(date)";   $VX analyze.py --work $W
echo "== DONE $(date)"
