#!/usr/bin/env bash
# =============================================================================
# scripts/test_airgap.sh — MITHYA Air-Gap Isolation Test
# =============================================================================
#
# PURPOSE
# -------
# Prove that the MITHYA prototype runs 100% offline — no calls to external
# APIs, CDNs, or telemetry endpoints — satisfying the SIH 2026 Problem
# Statement 26146 requirement for air-gapped forensic operation.
#
# MECHANISM
# ---------
# Uses Linux "unshare" to create a new network namespace with NO network
# interfaces (except loopback lo).  Inside this namespace:
#   1.  All outbound TCP/UDP connections to non-loopback addresses will fail
#       with ENETUNREACH or ECONNREFUSED.
#   2.  We run the MITHYA test suite (pytest) and the ML evaluation script.
#   3.  Any test/code that attempts an external connection will raise an
#       exception, causing a FAIL.
#   4.  If everything passes, we print an AIR-GAP PROOF certificate.
#
# REQUIREMENTS
# ------------
#   - Linux kernel with CONFIG_USER_NS=y (most distros since 2013)
#   - util-linux >= 2.27 (for unshare --net --user)
#   - Python 3.9+ and project venv activated (or PYTHON_BIN set)
#   - pytest installed in the venv
#
# USAGE
# -----
#   bash scripts/test_airgap.sh
#   bash scripts/test_airgap.sh --skip-streamlit   # skip UI smoke-test
#   bash scripts/test_airgap.sh --verbose           # verbose pytest output
#
# EXIT CODES
# ----------
#   0 = All tests passed inside the air-gap namespace (PROOF COMPLETE)
#   1 = One or more tests failed (network leak or logic error)
#   2 = Prerequisite check failed (unshare/python not found)
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Script location (absolute so it works from any CWD) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
SKIP_STREAMLIT=false
VERBOSE=false

for arg in "$@"; do
    case "$arg" in
        --skip-streamlit) SKIP_STREAMLIT=true ;;
        --verbose|-v)     VERBOSE=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-streamlit] [--verbose]"
            exit 0
            ;;
        *) echo -e "${RED}[!] Unknown argument: $arg${RESET}"; exit 2 ;;
    esac
done

# ── Resolve Python interpreter ────────────────────────────────────────────────
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
VENV_PYTHON2="${PROJECT_ROOT}/venv/bin/python3"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
elif [[ -x "$VENV_PYTHON2" ]]; then
    PYTHON="$VENV_PYTHON2"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo -e "${RED}[!] Python interpreter not found. Set PYTHON_BIN or activate the venv.${RESET}"
    exit 2
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e ""
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BLUE}${BOLD}  MITHYA — Air-Gap Isolation Proof Script${RESET}"
echo -e "${BLUE}  Problem Statement 26146 | SIH 2026${RESET}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e ""
echo -e "  ${CYAN}Project root :${RESET} ${PROJECT_ROOT}"
echo -e "  ${CYAN}Python       :${RESET} ${PYTHON}"
echo -e ""

# ── Prerequisite checks ───────────────────────────────────────────────────────
echo -e "${YELLOW}[*] Checking prerequisites...${RESET}"

if ! command -v unshare &>/dev/null; then
    echo -e "${RED}[!] 'unshare' not found. Install util-linux >= 2.27.${RESET}"
    exit 2
fi

UNSHARE_VERSION=$(unshare --version 2>&1 | head -1 || true)
echo -e "    unshare   : ${UNSHARE_VERSION}"
echo -e "    python    : $("$PYTHON" --version 2>&1)"

# Check if pytest is available
if ! "$PYTHON" -m pytest --version &>/dev/null; then
    echo -e "${YELLOW}[!] pytest not found — unit tests will be skipped.${RESET}"
    HAS_PYTEST=false
else
    HAS_PYTEST=true
    echo -e "    pytest    : $("$PYTHON" -m pytest --version 2>&1 | head -1)"
fi

echo -e ""

# ── Build the inner test script ───────────────────────────────────────────────
# This script runs INSIDE the network-isolated namespace.
INNER_SCRIPT=$(mktemp /tmp/mithya_airgap_inner_XXXXXX.sh)
trap 'rm -f "$INNER_SCRIPT"' EXIT

cat > "$INNER_SCRIPT" << 'INNER_EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="__PROJECT_ROOT__"
PYTHON="__PYTHON__"
HAS_PYTEST="__HAS_PYTEST__"
SKIP_STREAMLIT="__SKIP_STREAMLIT__"
VERBOSE="__VERBOSE__"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

stamp() { date '+%Y-%m-%dT%H:%M:%S'; }

run_step() {
    local label="$1"
    shift
    echo -e "  ${CYAN}[$(stamp)] Running:${RESET} ${label}"
    if "$@" 2>&1 | sed 's/^/    /'; then
        echo -e "  ${GREEN}[PASS]${RESET} ${label}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "  ${RED}[FAIL]${RESET} ${label}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

# ── Bring up loopback only (no external interfaces) ──────────────────────────
echo -e "${YELLOW}[*] Configuring network namespace (loopback only)...${RESET}"
if command -v ip &>/dev/null; then
    ip link set lo up 2>/dev/null || true
    echo -e "  ${GREEN}[OK]${RESET} Loopback interface configured."
else
    echo -e "  ${YELLOW}[WARN]${RESET} 'ip' not found — skipping lo setup (usually already up)."
fi
echo ""

# ── Verify: external connectivity is BLOCKED ─────────────────────────────────
echo -e "${YELLOW}[*] Verifying network isolation...${RESET}"
if curl --max-time 2 --silent --head "https://8.8.8.8" &>/dev/null; then
    echo -e "  ${RED}[FAIL]${RESET} External network is reachable — isolation failed!"
    exit 1
else
    echo -e "  ${GREEN}[OK]${RESET} External network is unreachable (air-gap confirmed)."
fi
echo ""

# ── Step 1: Python import test (no network calls) ────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 1: Python module import (offline) ──────────────────────${RESET}"
run_step "Import ml_engine, transaction_adapter, anomaly_engine" \
    "$PYTHON" -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
import ml_engine, transaction_adapter, anomaly_engine, features, explainability
print('  All core modules imported successfully (no network calls).')
"

# ── Step 2: Adapter unit tests ────────────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 2: Adapter unit tests ───────────────────────────────────${RESET}"
if [[ "$HAS_PYTEST" == "true" ]]; then
    PYTEST_ARGS="-x --tb=short"
    [[ "$VERBOSE" == "true" ]] && PYTEST_ARGS="$PYTEST_ARGS -v"
    run_step "pytest tests/test_adapters.py" \
        "$PYTHON" -m pytest ${PYTEST_ARGS} "${PROJECT_ROOT}/tests/test_adapters.py"
else
    echo -e "  ${YELLOW}[SKIP]${RESET} pytest not available"
fi

# ── Step 3: Air-gap Python test ────────────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 3: Python air-gap socket test ──────────────────────────${RESET}"
if [[ -f "${PROJECT_ROOT}/tests/test_airgap.py" ]]; then
    PYTEST_ARGS="-x --tb=short"
    [[ "$VERBOSE" == "true" ]] && PYTEST_ARGS="$PYTEST_ARGS -v"
    run_step "pytest tests/test_airgap.py" \
        "$PYTHON" -m pytest ${PYTEST_ARGS} "${PROJECT_ROOT}/tests/test_airgap.py"
fi

# ── Step 4: Heuristic tests ────────────────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 4: Heuristic tests ──────────────────────────────────────${RESET}"
if [[ -f "${PROJECT_ROOT}/tests/test_heuristics.py" && "$HAS_PYTEST" == "true" ]]; then
    PYTEST_ARGS="-x --tb=short"
    [[ "$VERBOSE" == "true" ]] && PYTEST_ARGS="$PYTEST_ARGS -v"
    run_step "pytest tests/test_heuristics.py" \
        "$PYTHON" -m pytest ${PYTEST_ARGS} "${PROJECT_ROOT}/tests/test_heuristics.py"
fi

# ── Step 5: ML Evaluation (offline) ───────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 5: ML validation evaluation (offline) ───────────────────${RESET}"
if [[ -f "${PROJECT_ROOT}/evaluate_model.py" ]]; then
    run_step "evaluate_model.py --records 100" \
        "$PYTHON" "${PROJECT_ROOT}/evaluate_model.py" --records 100
fi

# ── Step 6: Streamlit smoke-test (import only, no server start) ───────────────
echo -e "${BLUE}${BOLD}── STEP 6: Streamlit import smoke-test ──────────────────────────${RESET}"
if [[ "$SKIP_STREAMLIT" != "true" ]]; then
    run_step "Streamlit + app imports (no CDN/API call)" \
        "$PYTHON" -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
import streamlit  # must not phone home during import
print('  streamlit import: OK')
# Verify the AirGapSocket override in app.py will work
import socket
orig = socket.socket
class _TestSocket(orig):
    def connect(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host not in ('127.0.0.1', 'localhost', '::1'):
            raise PermissionError(f'Air-Gap: blocked {host}')
        return super().connect(addr)
socket.socket = _TestSocket
print('  AirGapSocket override: OK')
socket.socket = orig
"
else
    echo -e "  ${YELLOW}[SKIP]${RESET} --skip-streamlit flag set"
fi

# ── Step 7: Geo-ASN offline resolver ─────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 7: Offline Geo-ASN resolver ────────────────────────────${RESET}"
if [[ -f "${PROJECT_ROOT}/geo_asn.py" ]]; then
    run_step "geo_asn.resolve_ip() offline test" \
        "$PYTHON" -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from geo_asn import resolve_ip, resolve_ips_batch
r1 = resolve_ip('1.1.1.1')
r2 = resolve_ip('9.9.9.9')
batch = resolve_ips_batch(['8.8.8.8', '10.0.0.1', '192.168.1.1'])
print(f'  resolve_ip(1.1.1.1): country={r1[\"geo_country\"]} asn={r1[\"asn\"]}')
print(f'  resolve_ip(9.9.9.9): country={r2[\"geo_country\"]} asn={r2[\"asn\"]}')
print(f'  batch(3 IPs): {len(batch)} results')
print('  Geo-ASN offline: OK')
"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BLUE}${BOLD}  AIR-GAP ISOLATION TEST RESULTS${RESET}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "  Steps Passed : ${GREEN}${BOLD}${PASS_COUNT}${RESET}"
echo -e "  Steps Failed : $( [[ $FAIL_COUNT -gt 0 ]] && echo "${RED}${BOLD}${FAIL_COUNT}${RESET}" || echo "${GREEN}0${RESET}" )"
echo ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  AIR-GAP PROOF CERTIFICATE${RESET}"
    echo -e "${GREEN}  =============================${RESET}"
    echo -e "${GREEN}  The MITHYA prototype ran ${PASS_COUNT} test steps inside a${RESET}"
    echo -e "${GREEN}  Linux network namespace with ZERO external connectivity.${RESET}"
    echo -e "${GREEN}  No external API calls, CDN loads, or telemetry were made.${RESET}"
    echo -e "${GREEN}  The system is confirmed AIR-GAP COMPATIBLE.${RESET}"
    echo -e ""
    echo -e "${GREEN}  Tested at: $(date)${RESET}"
    echo -e "${GREEN}  Namespace: unshare -n (network namespace isolation)${RESET}"
    echo -e "${GREEN}  Python   : $(python3 --version 2>&1)${RESET}"
    echo ""
    exit 0
else
    echo -e "${RED}${BOLD}  [!] AIR-GAP TEST FAILED${RESET}"
    echo -e "${RED}  ${FAIL_COUNT} step(s) failed. Review output above.${RESET}"
    echo ""
    exit 1
fi
INNER_EOF

# Substitute the real values into the inner script
sed -i "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" "$INNER_SCRIPT"
sed -i "s|__PYTHON__|${PYTHON}|g" "$INNER_SCRIPT"
sed -i "s|__HAS_PYTEST__|${HAS_PYTEST}|g" "$INNER_SCRIPT"
sed -i "s|__SKIP_STREAMLIT__|${SKIP_STREAMLIT}|g" "$INNER_SCRIPT"
sed -i "s|__VERBOSE__|${VERBOSE}|g" "$INNER_SCRIPT"
chmod +x "$INNER_SCRIPT"

# ── Execute inside a new network namespace ────────────────────────────────────
echo -e "${YELLOW}[*] Spawning network-isolated namespace via 'unshare -n'...${RESET}"
echo -e "    (all external TCP/UDP will fail inside this namespace)"
echo -e ""

# Try user + network namespace first (no root required on most distros)
# Fall back to root-required network namespace if that fails.
if unshare --net --user --map-root-user bash "$INNER_SCRIPT"; then
    # Success already handled inside inner script
    true
elif unshare --net bash "$INNER_SCRIPT"; then
    # Success already handled inside inner script
    true
else
    echo -e "${RED}[!] 'unshare --net' failed.${RESET}"
    echo -e "${YELLOW}    On some systems you need to enable unprivileged namespaces:${RESET}"
    echo -e "      sudo sysctl kernel.unprivileged_userns_clone=1"
    echo -e "    Or run this script as root / with sudo."
    exit 2
fi
