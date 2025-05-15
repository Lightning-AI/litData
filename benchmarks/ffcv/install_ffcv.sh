echo "=== FFCV Installer ==="
OS="$(uname -s)"
echo "Detected OS: $OS"

# Install ffcv dependencies
conda install -y -c conda-forge libjpeg-turbo
conda install -y pkg-config compilers opencv -c conda-forge
pip uninstall -y opencv-python-headless numba
pip install opencv-python-headless numba
pip install --force-reinstall "numpy>=1.21,<2"
pip install ffcv

echo "Verifying FFCV installation..."
python3 -c "import ffcv; print('✅ FFCV installed successfully!')"

echo "=== Done ==="
