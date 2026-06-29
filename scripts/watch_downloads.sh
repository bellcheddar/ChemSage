#!/usr/bin/env bash
MODELS_DIR="/Users/dellboy/Documents/Vibe_Coding/chem_sage/models"
FILES=(
  "chem_sage_32b_v2/model-00002-of-00004.safetensors:5335712972"
  "chem_sage_32b_v2/model-00003-of-00004.safetensors:5366641944"
  "chem_sage_32b_v2/model-00004-of-00004.safetensors:2362540824"
  "chem_sage_32b_v3/model-00002-of-00004.safetensors:5335712972"
  "chem_sage_32b_v3/model-00003-of-00004.safetensors:5366641944"
  "chem_sage_32b_v3/model-00004-of-00004.safetensors:2362540824"
  "chem_sage_32b_v4/model-00002-of-00004.safetensors:5335712972"
  "chem_sage_32b_v4/model-00003-of-00004.safetensors:5366641944"
)
TOTAL=36869586532
START_TIME=$(date +%s)

while true; do
  clear
  DONE_BYTES=0
  echo "  ChemSage — iCloud Model Download Progress"
  echo "  $(date '+%H:%M:%S')"
  echo ""
  echo "  File                                        Status      Size"
  echo "  ──────────────────────────────────────────────────────────────"
  for entry in "${FILES[@]}"; do
    f="${entry%%:*}"; sz="${entry##*:}"
    path="$MODELS_DIR/$f"
    flag=$(ls -lO "$path" 2>/dev/null | awk '{print $5}')
    model="${f%%/*}"; short="${f##*/}"
    label="${model##*_v}/${short%%.*}"
    if [ "$flag" = "compressed,dataless" ]; then
      printf "  %-44s ⏳ cloud   %4.1f GB\n" "v${label}" "$(echo "scale=1; $sz/1073741824" | bc)"
    else
      DONE_BYTES=$((DONE_BYTES + sz))
      printf "  %-44s ✓ local   %4.1f GB\n" "v${label}" "$(echo "scale=1; $sz/1073741824" | bc)"
    fi
  done

  echo "  ──────────────────────────────────────────────────────────────"
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))
  DONE_GB=$(echo "scale=2; $DONE_BYTES / 1073741824" | bc)
  TOT_GB=$(echo "scale=2; $TOTAL / 1073741824" | bc)
  PCT=$(echo "scale=1; $DONE_BYTES * 100 / $TOTAL" | bc)
  FILLED=$(echo "$DONE_BYTES * 40 / $TOTAL" | bc)
  BAR=$(printf "%${FILLED}s" | tr " " "█")$(printf "%$((40 - FILLED))s" | tr " " "░")

  if [ "$DONE_BYTES" -gt 0 ] && [ "$ELAPSED" -gt 0 ]; then
    RATE=$(echo "scale=0; $DONE_BYTES / $ELAPSED" | bc)
    REMAIN=$((TOTAL - DONE_BYTES))
    ETA_S=$(echo "scale=0; $REMAIN / $RATE" | bc 2>/dev/null)
    ETA_M=$((ETA_S / 60)); ETA_S2=$((ETA_S % 60))
    ETA_STR="${ETA_M}m ${ETA_S2}s"
    RATE_MB=$(echo "scale=1; $RATE / 1048576" | bc)
  else
    ETA_STR="calculating..."
    RATE_MB="0"
  fi

  echo ""
  echo "  [${BAR}] ${PCT}%"
  echo "  ${DONE_GB} / ${TOT_GB} GB    ${RATE_MB} MB/s    ETA: ${ETA_STR}"
  echo ""

  if [ "$DONE_BYTES" -eq "$TOTAL" ]; then
    echo "  All files downloaded. Ready to run eval."
    exit 0
  fi

  sleep 10
done
