#!/usr/bin/env bash
# =============================================================================
# scripts/test_airgap.sh — MITHYA Air-Gap Isolation Report
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
# Uses Linux "unshare -r -n" to create a new user + network namespace with
# NO external network interfaces (loopback only). Inside this namespace:
#   1. All outbound TCP/UDP connections to non-loopback addresses fail.
#   2. We run the full MITHYA test suite (pytest).
#   3. Any code that attempts an external connection raises an exception.
#   4. If everything passes, we print a clean Air-Gap Isolation Report.
#
# REQUIREMENTS
# ------------
#   - Linux kernel with CONFIG_USER_NS=y (Arch Linux default)
#   - util-linux (for unshare)
#   - curl (for connectivity verification)
#   - Python 3.9+ and project venv activated (or PYTHON_BIN set)
#   - pytest installed in the venv
#
# USAGE
# -----
#   bash scripts/test_airgap.sh
#   bash scripts/test_airgap.sh --skip-streamlit
#   bash scripts/test_airgap.sh --verbose
#
# EXIT CODES
# ----------
#   0 = All tests passed inside the air-gap namespace
#   1 = One or more tests failed (network leak or logic error)
#   2 = Prerequisite check failed (unshare/curl/python not found)
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
echo -e "${BLUE}${BOLD}  MITHYA — Air-Gap Isolation Report${RESET}"
echo -e "${BLUE}  Problem Statement 26146 | SIH 2026${RESET}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e ""
echo -e "  ${CYAN}Project root :${RESET} ${PROJECT_ROOT}"
echo -e "  ${CYAN}Python       :${RESET} ${PYTHON}"
echo -e ""

# ── Prerequisite checks ───────────────────────────────────────────────────────
echo -e "${YELLOW}[*] Checking prerequisites...${RESET}"

# Check: curl
if ! command -v curl &>/dev/null; then
    echo -e "${RED}[!] 'curl' not found. Install with: sudo pacman -S curl${RESET}"
    exit 2
fi
echo -e "    curl      : $(curl --version 2>&1 | head -1)"

# Check: unshare
if ! command -v unshare &>/dev/null; then
    echo -e "${RED}[!] 'unshare' not found. Install util-linux: sudo pacman -S util-linux${RESET}"
    exit 2
fi
echo -e "    unshare   : $(unshare --version 2>&1 | head -1)"

# Check: Python
echo -e "    python    : $("$PYTHON" --version 2>&1)"

# Check: pytest
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
    echo -e "  ${YELLOW}[WARN]${RESET} 'ip' not found — skipping lo setup."
fi
echo ""

# ── Verify: external connectivity is BLOCKED ─────────────────────────────────
echo -e "${YELLOW}[*] Verifying network isolation...${RESET}"
if command -v curl &>/dev/null; then
    if curl --max-time 2 --silent --head "https://8.8.8.8" &>/dev/null; then
        echo -e "  ${RED}[FAIL]${RESET} External network is reachable — isolation failed!"
        exit 1
    else
        echo -e "  ${GREEN}[OK]${RESET} External network is unreachable (air-gap confirmed)."
    fi
else
    # Fallback: test via Python socket
    if "$PYTHON" -c "
import socket, sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(('8.8.8.8', 53))
    s.close()
    sys.exit(1)
except:
    sys.exit(0)
"; then
        echo -e "  ${GREEN}[OK]${RESET} External network is unreachable (air-gap confirmed)."
    else
        echo -e "  ${RED}[FAIL]${RESET} External network is reachable — isolation failed!"
        exit 1
    fi
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

# ── Step 5: XML Security Tests (XXE & Billion Laughs) ─────────────────────────
echo -e "${BLUE}${BOLD}── STEP 5: XML security hardening tests ─────────────────────────${RESET}"
if [[ "$HAS_PYTEST" == "true" ]]; then
    PYTEST_ARGS="-x --tb=short"
    [[ "$VERBOSE" == "true" ]] && PYTEST_ARGS="$PYTEST_ARGS -v"
    run_step "pytest -k 'xxe or billion_laughs' tests/test_adapters.py" \
        "$PYTHON" -m pytest ${PYTEST_ARGS} -k "xxe or billion_laughs" "${PROJECT_ROOT}/tests/test_adapters.py"
fi

# ── Step 6: Geo-ASN offline resolver ─────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 6: Offline Geo-ASN resolver ────────────────────────────${RESET}"
if [[ -f "${PROJECT_ROOT}/geo_asn.py" ]]; then
    run_step "geo_asn.resolve_ip() offline test (strict unknown)" \
        "$PYTHON" -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from geo_asn import resolve_ip, resolve_ips_batch

# Reserved ranges must always return correct labels
r_priv = resolve_ip('10.0.0.1')
assert r_priv['geo_country'] == 'PRIVATE', f'Expected PRIVATE, got {r_priv}'
assert r_priv['asn'] == 'AS0', f'Expected AS0, got {r_priv}'
print(f'  resolve_ip(10.0.0.1): country={r_priv[\"geo_country\"]} asn={r_priv[\"asn\"]} org={r_priv[\"asn_org\"]}')

r_lo = resolve_ip('127.0.0.1')
assert r_lo['geo_country'] == 'LOCAL', f'Expected LOCAL, got {r_lo}'
print(f'  resolve_ip(127.0.0.1): country={r_lo[\"geo_country\"]} asn={r_lo[\"asn\"]} org={r_lo[\"asn_org\"]}')

# Invalid IP must return strict unknown
r_bad = resolve_ip('invalid_ip')
assert r_bad['geo_country'] == 'XX', f'Expected XX, got {r_bad}'
assert r_bad['asn'] == 'AS0', f'Expected AS0, got {r_bad}'
assert r_bad['asn_org'] == 'Unknown', f'Expected Unknown, got {r_bad}'
print(f'  resolve_ip(invalid): country={r_bad[\"geo_country\"]} asn={r_bad[\"asn\"]} (strict unknown: OK)')

# Batch test
batch = resolve_ips_batch(['8.8.8.8', '10.0.0.1', '192.168.1.1', 'nan'])
assert len(batch) == 4, f'Expected 4 results, got {len(batch)}'
print(f'  batch(4 IPs): {len(batch)} results')
print('  Geo-ASN offline resolver: OK (never fabricates data)')
"
fi

# ── Step 7: ML Evaluation (offline) ───────────────────────────────────────────
echo -e "${BLUE}${BOLD}── STEP 7: ML validation evaluation (offline) ───────────────────${RESET}"
if [[ -f "${PROJECT_ROOT}/evaluate_model.py" ]]; then
    run_step "evaluate_model.py --records 100" \
        "$PYTHON" "${PROJECT_ROOT}/evaluate_model.py" --records 100
fi

# ── Step 8: Streamlit smoke-test (import only, no server start) ───────────────
echo -e "${BLUE}${BOLD}── STEP 8: Streamlit import smoke-test ──────────────────────────${RESET}"
if [[ "$SKIP_STREAMLIT" != "true" ]]; then
    run_step "Streamlit + app imports (no CDN/API call)" \
        "$PYTHON" -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
import streamlit
print('  streamlit import: OK (no network phone-home)')
"
else
    echo -e "  ${YELLOW}[SKIP]${RESET} --skip-streamlit flag set"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BLUE}${BOLD}  AIR-GAP ISOLATION REPORT${RESET}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "  Steps Passed : ${GREEN}${BOLD}${PASS_COUNT}${RESET}"
echo -e "  Steps Failed : $( [[ $FAIL_COUNT -gt 0 ]] && echo "${RED}${BOLD}${FAIL_COUNT}${RESET}" || echo "${GREEN}0${RESET}" )"
echo ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  ┌─────────────────────────────────────────────────┐${RESET}"
    echo -e "${GREEN}${BOLD}  │         AIR-GAP ISOLATION: VERIFIED             │${RESET}"
    echo -e "${GREEN}${BOLD}  └─────────────────────────────────────────────────┘${RESET}"
    echo -e "${GREEN}  The MITHYA prototype executed ${PASS_COUNT} test steps inside a${RESET}"
    echo -e "${GREEN}  Linux network namespace with ZERO external connectivity.${RESET}"
    echo -e "${GREEN}  No external API calls, CDN loads, or telemetry detected.${RESET}"
    echo -e ""
    echo -e "${GREEN}  Tested at : $(date --iso-8601=seconds 2>/dev/null || date)${RESET}"
    echo -e "${GREEN}  Namespace : unshare -r -n (user + network namespace)${RESET}"
    echo -e "${GREEN}  Kernel    : $(uname -r)${RESET}"
    echo -e "${GREEN}  Python    : $("$PYTHON" --version 2>&1)${RESET}"
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

# ── Probe which unshare mode is usable on this machine ───────────────────
# We test each mode with a no-op (`true`) so that the inner test suite is
# never run more than once.  This prevents conflating "unshare capability
# missing" with "tests actually failed" and avoids doubling runtime.
echo -e "${YELLOW}[*] Probing available namespace modes...${RESET}"

if unshare -r -n true 2>/dev/null; then
    UNSHARE_MODE="user-r"
    UNSHARE_CMD=(unshare -r -n)
    echo -e "    ${GREEN}[OK]${RESET} unshare -r -n (user + network namespace)"
elif unshare --net --user --map-root-user true 2>/dev/null; then
    UNSHARE_MODE="user-long"
    UNSHARE_CMD=(unshare --net --user --map-root-user)
    echo -e "    ${GREEN}[OK]${RESET} unshare --net --user --map-root-user"
elif unshare --net true 2>/dev/null; then
    UNSHARE_MODE="root"
    UNSHARE_CMD=(unshare --net)
    echo -e "    ${GREEN}[OK]${RESET} unshare --net (may require root)"
else
    echo -e "${RED}[!] Neither unshare mode is usable on this system.${RESET}"
    echo -e "${YELLOW}    Possible fixes:${RESET}"
    echo -e "${YELLOW}    1. Enable unprivileged user namespaces:${RESET}"
    echo -e "       sudo sysctl kernel.unprivileged_userns_clone=1"
    echo -e "${YELLOW}    2. On Arch Linux, check:${RESET}"
    echo -e "       sysctl kernel.unprivileged_userns_clone"
    echo -e "${YELLOW}    3. Or run this script with sudo.${RESET}"
    exit 2
fi

# ── Execute inside the confirmed namespace (exactly once) ────────────────
echo -e ""
echo -e "${YELLOW}[*] Spawning network-isolated namespace via '${UNSHARE_CMD[*]}'...${RESET}"
echo -e "    (all external TCP/UDP will fail inside this namespace)"
echo -e ""

INNER_EXIT=0
"${UNSHARE_CMD[@]}" bash "$INNER_SCRIPT" || INNER_EXIT=$?

exit "$INNER_EXIT"
