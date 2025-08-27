# Complete, reproducible script to build and prepare environment

# should probably allocate a node to work on like so since we need the cpu cores for the build
# flux alloc -q pbatch --bank=guests --job-name=build -t240 -N1 -n1 -g1 -c96 -ofastload -o mpibind=off --exclusive --unbuffered --label-io

CURR_REPO=$(pwd)

# modify the installation path and env name if you want
INSTALLDIR=${WRKSPC}
ENV_NAME="tuolumne_conda_28_630_mdpo"

cd ${INSTALLDIR}

# Base the installation on previously installed miniconda.
# Note, this is a manual process currently.

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.12 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
conda install -c conda-forge conda-pack libstdcxx-ng --yes

# Load modules
rocm_version=6.3.0

module load rocm/$rocm_version
module load gcc-native/12.2

######### COMPILE PIP PACKAGES ########################

# pytorch and core reqs
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install torch==2.8.0+rocm6.3 --index-url https://download.pytorch.org/whl/rocm6.3
pip install ninja packaging numpy

cd "${CURR_REPO}"
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install -e .
cd ${INSTALLDIR}

#### peft ####
# for some reason needed to install peft afterward manually? not sure why
pip install "peft>=0.14.0"

#### bnb ####
# saw a runtime warning for ROCM libraries

# pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-1.0.0-py3-none-manylinux_2_24_x86_64.whl' --no-deps
# wheel doesnt work

# Install bitsandbytes from source
# Clone bitsandbytes repo, ROCm backend is currently enabled on multi-backend-refactor branch
# git clone -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git
# neither does this but that's bc its the same potentially

# but this works!
# https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1519
git clone -b multi-backend-refactor-a0a95fd-hacky-fix https://github.com/xzuyn/bitsandbytes.git
cd bitsandbytes/

# Compile & install
# apt-get install -y build-essential cmake  # install build tools dependencies, unless present
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S .  # Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch
make
pip install --no-deps -e .   # `-e` for "editable" install, when developing BNB (otherwise leave that out)


# amdsmi
cp -R /opt/rocm-${rocm_version}/share/amd_smi/ $WRKSPC/amd_smi_${rocm_version}
cd $WRKSPC/amd_smi_${rocm_version}
pip install .
cd ${INSTALLDIR}