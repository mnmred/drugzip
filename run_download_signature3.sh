#!/bin/bash
#SBATCH --job-name=cc-sign3-dl
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --chdir=/tudelft.net/staff-umbrella/nbrouwer1/drugzip
#SBATCH --output=logs/cc_sign3_dl_%j.out
#SBATCH --error=logs/cc_sign3_dl_%j.err

# Download Chemical Checker signature3 HDF5 files (~25 x 700 MB ≈ 17 GB).
# Source: https://chemicalchecker.com/downloads/signature3
#
# Prefer the login node (compute nodes often lack outbound HTTPS):
#   cd /tudelft.net/staff-umbrella/nbrouwer1/drugzip
#   bash run_download_signature3.sh
#
# Or: sbatch run_download_signature3.sh
# Override: DRUGZIP_ROOT / OUT_DIR

set -euo pipefail

# Do NOT reuse ambient PROJECTDIR — on this cluster it is often the umbrella
# parent (…/nbrouwer1), not the drugzip repo.
DRUGZIP_ROOT="${DRUGZIP_ROOT:-/tudelft.net/staff-umbrella/nbrouwer1/drugzip}"
OUT_DIR="${OUT_DIR:-${DRUGZIP_ROOT}/Data/ChemicalChecker_Full/h5}"
BASE_URL="https://chemicalchecker.com/api/db/getFile/signature3"

FILES=(
  A1_sign3.h5 A2_sign3.h5 A3_sign3.h5 A4_sign3.h5 A5_sign3.h5
  B1_sign3.h5 B2_sign3.h5 B3_sign3.h5 B4_sign3.h5 B5_sign3.h5
  C1_sign3.h5 C2_sign3.h5 C3_sign3.h5 C4_sign3.h5 C5_sign3.h5
  D1_sign3.h5 D2_sign3.h5 D3_sign3.h5 D4_sign3.h5 D5_sign3.h5
  E1_sign3.h5 E2_sign3.h5 E3_sign3.h5 E4_sign3.h5 E5_sign3.h5
)

# Flags compatible with older RHEL/CentOS curl on the login nodes
curl_get() {
  curl -fL --retry 8 --retry-delay 15 --connect-timeout 30 \
    -A "Mozilla/5.0 (compatible; DrugZip-CC-download/1.0)" \
    "$@"
}

mkdir -p "$OUT_DIR" "${DRUGZIP_ROOT}/logs"
cd "$OUT_DIR"

echo "Host: $(hostname)"
echo "DRUGZIP_ROOT=${DRUGZIP_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "curl: $(curl --version | head -1)"
echo "Will download ${#FILES[@]} files"

probe_url="${BASE_URL}/${FILES[0]}/"
echo "Probing ${probe_url} ..."
if ! curl_get -I "$probe_url" >/tmp/cc_sign3_probe_$$.hdr 2>/tmp/cc_sign3_probe_$$.err; then
  echo "ERROR: HTTPS to chemicalchecker.com failed on $(hostname)." >&2
  cat /tmp/cc_sign3_probe_$$.err >&2 || true
  echo "Try from login node: cd ${DRUGZIP_ROOT} && bash run_download_signature3.sh" >&2
  exit 1
fi
rm -f /tmp/cc_sign3_probe_$$.hdr /tmp/cc_sign3_probe_$$.err
echo "Probe OK."

for name in "${FILES[@]}"; do
  dest="${OUT_DIR}/${name}"
  url="${BASE_URL}/${name}/"

  if [ -f "$dest" ]; then
    remote_size="$(curl_get -sI "$url" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r' | tail -1)"
    local_size="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if [ -n "${remote_size:-}" ] && [ "$local_size" = "$remote_size" ]; then
      echo "SKIP (complete): ${name} (${local_size} bytes)"
      continue
    fi
    echo "RESUME: ${name} (local=${local_size} remote=${remote_size:-unknown})"
  else
    echo "GET: ${name}"
  fi

  curl_get -C - -o "$dest" "$url"
  echo "DONE: ${name} ($(stat -c%s "$dest") bytes)"
done

echo
echo "All downloads finished. Files in ${OUT_DIR}:"
ls -lh "$OUT_DIR"/*.h5

echo
echo "Next: convert to CSVs expected by ae.py (A1.csv..E5.csv):"
echo "  python ${DRUGZIP_ROOT}/csv_signature_generator.py ${OUT_DIR} ${DRUGZIP_ROOT}/Data/ChemicalChecker_Full"
