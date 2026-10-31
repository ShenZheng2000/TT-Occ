#!/bin/bash
# Usage: bash run.sh configs/<name>.env [NUM_SCENES=8] [NUM_GPUS=8]
# = run_parallel.sh + run_summary.sh, settings from a config, output to $WORK_DIRS/<name>/<timestamp>/.
# Scores: <run_dir>/results-Occ3D/results.json, <run_dir>/results-nuCraft/results.json

cd "$(dirname "${BASH_SOURCE[0]}")"
CFG=$1; N=${2:-8}; G=${3:-8}
set -a; source "$CFG"; set +a

export OUTPUT_ROOT=$WORK_DIRS/$(basename "$CFG" .env)/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_ROOT/logs"
bash -c "set -a; source $CFG; env" | grep -f <(grep -oh '^[A-Z0-9_]*=' configs/base.env "$CFG") > "$OUTPUT_ROOT/config.env"
echo "RUN_DIR: $OUTPUT_ROOT"

# run_parallel.sh
mapfile -t ALL < <(grep -o 'scene-[0-9]\+' run_main.sh | awk '!s[$0]++' | head -n "$N")
for ((g=0; g<G; g++)); do
    chunk=""
    for ((i=g; i<${#ALL[@]}; i+=G)); do chunk+="${ALL[$i]} "; done
    [ -z "$chunk" ] && continue
    echo "GPU $g: $chunk"
    CUDA_VISIBLE_DEVICES=$g SCENES="$chunk" bash run_main.sh > "$OUTPUT_ROOT/logs/gpu$g.log" 2>&1 &
done
wait

# run_summary.sh
python summarymiou.py Occ3D "$OUTPUT_ROOT/results-Occ3D" "$DATA_ROOT" "$OCC3D_PATH" > "$OUTPUT_ROOT/logs/summary_occ3d.log" 2>&1
python summarymiou.py nuCraft "$OUTPUT_ROOT/results-nuCraft" "$DATA_ROOT" "$NUCRAFT_PATH" > "$OUTPUT_ROOT/logs/summary_nucraft.log" 2>&1
echo "all done: $OUTPUT_ROOT"
