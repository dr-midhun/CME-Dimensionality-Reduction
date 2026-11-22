#!/usr/bin/env bash
# Setup prerequisites for macOS and Linux
set -euo pipefail

OS="$(uname -s)"

echo "==> Setting up prerequisites for $OS..."

if [ "$OS" = "Darwin" ]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Error: Homebrew is required on macOS. Install it from https://brew.sh/"
        exit 1
    fi
    echo "==> Installing libomp and cmake via Homebrew..."
    brew install libomp cmake
    echo "==> Done. You can now run 'make' or 'make test'."

elif [ "$OS" = "Linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "==> Detected Debian/Ubuntu system."
        echo "==> Running apt-get update and installing build-essential, libomp-dev, cmake..."
        sudo apt-get update
        sudo apt-get install -y build-essential libomp-dev cmake liblapack-dev
    elif command -v dnf >/dev/null 2>&1; then
        echo "==> Detected Fedora/RHEL system."
        sudo dnf install -y gcc-c++ make cmake libomp-devel lapack-devel
    elif command -v pacman >/dev/null 2>&1; then
        echo "==> Detected Arch Linux system."
        sudo pacman -S --needed base-devel cmake openmp lapack
    else
        echo "Warning: Unknown package manager. Please ensure g++, make, OpenMP, and cmake are installed."
    fi
    echo "==> Done. You can now run 'make' or 'make test'."
else
    echo "Unsupported OS: $OS"
    exit 1
fi
