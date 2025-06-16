#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# Target frequency in kHz (600 MHz)
FREQ_KHZ=600000
# Which CPU to pin everything to
CPU=0
# List of binaries you want to run
BINS=(
  "baseline_lenet_small"
  "baseline_lenet_large"
  "baseline_vgg16"
)
# Directory to dump logs
LOGDIR="result"
mkdir -p "${LOGDIR}"

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────
set_freq() {
  echo "[*] Setting CPU governor to userspace on cpu${CPU}…"
  echo userspace | sudo tee "/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_governor" >/dev/null
  echo "[*] Locking cpu${CPU} to ${FREQ_KHZ} kHz…"
  echo "${FREQ_KHZ}" | sudo tee "/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_setspeed" >/dev/null
}

run_bin() {
  local bin="$1"
  local logfile="${LOGDIR}/${bin}.log"
  echo "[+] Running ${bin} → ${logfile}"
  # pin to CPU and run in release mode
  taskset -c "${CPU}" \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    cargo run --bin "${bin}" --release \
    > "${logfile}" 2>&1
}

# ─── MAIN ──────────────────────────────────────────────────────────────────────
echo "[*] Underclocking and pinning to CPU${CPU} at ${FREQ_KHZ} kHz"
set_freq

for bin in "${BINS[@]}"; do
  run_bin "$bin"
done

echo "[✔] All workloads complete. Logs in ${LOGDIR}/"
