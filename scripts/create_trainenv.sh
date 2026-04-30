#!/usr/bin/env bash
###############################################################################
# create_trainenv.sh — Automated conda training environment builder
#
# Features:
#   - Detects CUDA driver version from nvidia-smi
#   - Selects best matching pytorch-cuda toolkit version
#   - Installs latest stable Python 3.12.x (conda resolves best patch)
#   - Installs PyTorch, FLA, causal-conv1d, and all training deps
#   - Validates GPU compute capability for TORCH_CUDA_ARCH_LIST
#
# Usage:
#   chmod +x create_trainenv.sh
#   ./create_trainenv.sh [ENV_NAME]       # default: trainenv
#
# Josef's rig: RTX 3090 (sm_86) + RTX 3060 (sm_86) → TORCH_CUDA_ARCH_LIST="8.6"
###############################################################################
set -euo pipefail

ENV_NAME="${1:-trainenv}"
PYTHON_MAJOR_MINOR="3.12"   # Target Python series (conda picks latest patch)

# ANSI colors
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
CYN='\033[0;36m'
RST='\033[0m'

info()  { echo -e "${CYN}[INFO]${RST}  $*"; }
ok()    { echo -e "${GRN}[OK]${RST}    $*"; }
warn()  { echo -e "${YLW}[WARN]${RST}  $*"; }
die()   { echo -e "${RED}[FAIL]${RST}  $*" >&2; exit 1; }

###############################################################################
# 1. Pre-flight checks
###############################################################################
info "Running pre-flight checks..."

command -v conda  >/dev/null 2>&1 || die "conda not found. Install Miniconda/Anaconda first."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. NVIDIA drivers missing?"

# Auto-accept conda channel ToS if needed (required since conda 24.9+)
if conda tos status 2>&1 | grep -qi "not.*accepted\|pending\|need"; then
    info "Accepting conda channel Terms of Service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    ok "Conda ToS accepted."
else
    # Fallback: just try accepting anyway (the status command format varies by version)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

###############################################################################
# 2. Detect CUDA driver version from nvidia-smi
###############################################################################
info "Detecting CUDA driver version from nvidia-smi..."

# nvidia-smi top-right corner shows "CUDA Version: XX.Y"
CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
CUDA_VER_RAW=$(nvidia-smi 2>/dev/null \
    | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' \
    | head -1)

if [[ -z "${CUDA_VER_RAW:-}" ]]; then
    die "Could not parse CUDA version from nvidia-smi output."
fi

CUDA_MAJOR=$(echo "$CUDA_VER_RAW" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER_RAW" | cut -d. -f2)

ok "nvidia-smi reports CUDA driver capability: ${CUDA_VER_RAW}  (driver: ${CUDA_DRIVER_VER})"

###############################################################################
# 3. Map driver CUDA version → best pytorch-cuda toolkit version
#    PyTorch ships wheels for specific CUDA toolkits. We pick the highest
#    toolkit version that doesn't exceed the driver's capability.
#    As of mid-2025, PyTorch stable ships: cu118, cu121, cu124, cu126
###############################################################################
info "Selecting best PyTorch CUDA toolkit match..."

# Available pytorch-cuda versions (ascending). Update as PyTorch adds more.
PYTORCH_CUDA_OPTIONS=("11.8" "12.1" "12.4" "12.6")

SELECTED_CUDA=""
for opt in "${PYTORCH_CUDA_OPTIONS[@]}"; do
    opt_major=$(echo "$opt" | cut -d. -f1)
    opt_minor=$(echo "$opt" | cut -d. -f2)
    # Select if driver >= toolkit option
    if (( opt_major < CUDA_MAJOR )) || \
       (( opt_major == CUDA_MAJOR && opt_minor <= CUDA_MINOR )); then
        SELECTED_CUDA="$opt"
    fi
done

if [[ -z "$SELECTED_CUDA" ]]; then
    die "No compatible pytorch-cuda version found for driver CUDA ${CUDA_VER_RAW}."
fi

# Build the short tag (e.g., "12.4" → "cu124")
CUDA_TAG="cu$(echo "$SELECTED_CUDA" | tr -d '.')"

ok "Selected pytorch-cuda=${SELECTED_CUDA}  (wheel tag: ${CUDA_TAG})"

###############################################################################
# 4. Detect GPU compute capabilities for TORCH_CUDA_ARCH_LIST
###############################################################################
info "Detecting GPU compute capabilities..."

# Query all GPUs, deduplicate
ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
    | sort -uV \
    | paste -sd ';' -)

if [[ -z "$ARCH_LIST" ]]; then
    warn "Could not detect compute caps — defaulting to 8.6"
    ARCH_LIST="8.6"
fi

# Convert semicolons to spaces for display, semicolons for torch
ARCH_DISPLAY=$(echo "$ARCH_LIST" | tr ';' ' ')
TORCH_ARCH=$(echo "$ARCH_LIST" | tr ';' ' ')  # torch wants space or semicolon separated

ok "GPU compute capabilities: ${ARCH_DISPLAY}"

###############################################################################
# 5. Check if env already exists
###############################################################################
if conda env list 2>/dev/null | grep -qw "^${ENV_NAME} "; then
    warn "Conda environment '${ENV_NAME}' already exists."
    read -rp "Remove and recreate? [y/N] " ans
    if [[ "${ans,,}" == "y" ]]; then
        info "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        die "Aborted. Rename or remove the existing env first."
    fi
fi

###############################################################################
# 6. Create conda environment with best Python
###############################################################################
info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_MAJOR_MINOR}..."
conda create -n "$ENV_NAME" "python>=${PYTHON_MAJOR_MINOR},<3.13" -y

# Activate within this script's subshell
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

PYTHON_ACTUAL=$(python --version 2>&1)
ok "Environment created — ${PYTHON_ACTUAL}"

###############################################################################
# 7. Install PyTorch ecosystem (conda channel, matched CUDA)
###############################################################################
info "Installing PyTorch + torchvision + torchaudio (cuda=${SELECTED_CUDA})..."
conda install -y pytorch torchvision torchaudio "pytorch-cuda=${SELECTED_CUDA}" \
    -c pytorch -c nvidia

###############################################################################
# 9. Install core training dependencies
###############################################################################
info "Installing core training packages..."
conda install -y conda-forge::sentencepiece

pip install torchao
pip install bitsandbytes
pip install transformers        # For AutoTokenizer
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
pip install numpy
pip install zstandard
pip install sentencepiece

###############################################################################
# 10. Install causal-conv1d (needs TORCH_CUDA_ARCH_LIST)
###############################################################################
info "Installing causal-conv1d (TORCH_CUDA_ARCH_LIST=\"${TORCH_ARCH}\")..."
TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}" pip install "causal-conv1d>=1.4.0" --no-build-isolation

###############################################################################
# 11. Install flash-linear-attention (FLA)
#     Triton-based linear attention kernels — speeds up GDN / linear attn layers
#     Repo: https://github.com/sustcsonglin/flash-linear-attention
###############################################################################
info "Installing flash-linear-attention (FLA)..."
pip install flash-linear-attention --no-build-isolation

###############################################################################
# 12. Validation
###############################################################################
echo ""
info "Running post-install validation..."
echo "─────────────────────────────────────────────────────────"

validate() {
    local pkg="$1"
    local import_name="${2:-$1}"
    if python -c "import ${import_name}; print(f'  ${pkg}: {${import_name}.__version__}')" 2>/dev/null; then
        return 0
    else
        warn "  ${pkg}: import failed!"
        return 1
    fi
}

FAILURES=0

validate "torch"          "torch"           || ((FAILURES++))
validate "torchvision"    "torchvision"     || ((FAILURES++))
validate "torchaudio"     "torchaudio"      || ((FAILURES++))
validate "bitsandbytes"   "bitsandbytes"    || ((FAILURES++))
validate "transformers"   "transformers"    || ((FAILURES++))
validate "torchao"        "torchao"         || ((FAILURES++))
validate "sentencepiece"  "sentencepiece"   || ((FAILURES++))
validate "causal_conv1d"  "causal_conv1d"   || ((FAILURES++))
validate "fla"            "fla"             || ((FAILURES++))

# PyTorch CUDA check
echo ""
python -c "
import torch
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    dev = torch.cuda.get_device_name(0)
    cu_ver = torch.version.cuda
    print(f'  CUDA available: {dev} (toolkit {cu_ver})')
else:
    print('  WARNING: torch.cuda.is_available() = False!')
"

echo "─────────────────────────────────────────────────────────"

if (( FAILURES > 0 )); then
    warn "${FAILURES} package(s) failed validation — check output above."
else
    ok "All packages validated successfully!"
fi

echo ""
ok "Environment '${ENV_NAME}' is ready."
info "Activate with:  conda activate ${ENV_NAME}"
echo ""
