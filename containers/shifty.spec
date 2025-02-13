Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
mkdir $APPTAINER_ROOTFS/crispy_shifty/
cp -r /projects/crispy_shifty/crispy_shifty/ $APPTAINER_ROOTFS/crispy_shifty/crispy_shifty/
cp -r /projects/crispy_shifty/proteinmpnn/ $APPTAINER_ROOTFS/crispy_shifty/proteinmpnn/
cp -r /projects/crispy_shifty/proteininpainting/ $APPTAINER_ROOTFS/crispy_shifty/proteininpainting/
cp -r /projects/crispy_shifty/superfold/ $APPTAINER_ROOTFS/crispy_shifty/superfold/

%files
/etc/localtime
/etc/apt/sources.list
/etc/ssl/certs/ca-certificates.crt
/home/cdemakis/apptainer/files/bin/micromamba /opt/micromamba
#/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh 

%post
rm /bin/sh; ln -s /bin/bash /bin/sh

ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
apt-get install -y libx11-6 libxau6 libxext6 libxrender1 libgl1-mesa-glx
apt-get install -y git build-essential
apt-get install -y vim
apt-get clean

#bash /opt/miniconda.sh -b -u -p /usr
#rm /opt/miniconda.sh
rm -rf /usr/lib/terminfo
export MAMBA_ROOT_PREFIX=/usr
export MAMBA_EXE="/opt/micromamba";
eval "$(/opt/micromamba shell hook -s posix)"
export CONDA_OVERRIDE_CUDA=12

micromamba install -p /usr \
    -c pyg \
    -c pytorch \
    -c schrodinger \
    -c dglteam/label/cu117 \
    -c https://pose:foldtree@conda.rosettacommons.org \
    -c conda-forge \
    -c bioconda \
    -c nvidia \
    abseil-cpp \
    absl-py \
    atk-1.0 \
    attrs \
    billiard \
    binutils_impl_linux-64 \
    biopandas \
    biopython \
    black \
    blas \
    blosc \
    bokeh \
    chex \
    clang-11 \
    conda \
    conda-package-handling \
    contextlib2 \
    cryptography \
    cudatoolkit \
    cudnn \
    dask \
    dask-core \
    dask-jobqueue \
    dask-labextension \
    dataclasses \
    decorator \
    dgl \
    dglteam::dgl-cuda11.7 \
    distributed \
    dm-tree \
    dnachisel \
    flametree \
    flask \
    flatbuffers \
    git \
    gitpython \
    graphviz \
    h5py \
    holoviews \
    icecream \
    idna \
    immutabledict \
    ipykernel \
    ipympl \
    ipython \
    ipython_genutils \
    ipywidgets \
    isort \
    jax \
    jaxlib=*=*cuda*py38* \
    jupyter \
    jupyter-server-mathjax \
    jupyter-server-proxy \
    jupyter_client \
    jupyter_console \
    jupyter_core \
    jupyter_server \
    jupyterlab \
    jupyterlab-fasta \
    jupyterlab-git \
    jupyterlab_pygments \
    jupyterlab_server \
    jupyterlab_widgets \
    keras \
    lmfit \
    logomaker \
    mamba \
    markdown \
    matplotlib \
    mock \
    more-itertools \
    nb_black \
    nbclassic \
    nbclient \
    nbconvert \
    nbdime \
    nbformat \
    ncurses \
    networkx=2.8.4 \
    nglview \
    nodejs \
    numba \
    numdifftools \
    numpy=1.23.5 \
    nvidia-apex \
    openbabel \
    openmm=7.5.1 \
    openpyxl \
    pandas \
    pandoc \
    parallel \
    perl \
    pip \
    plotly \
    proglog \
    psutil \
    py3dmol \
    pybind11 \
    pyg=*=*cu* \
    pymatgen \
    pymol \
    pymol-bundle \
    pynvim \
    pynvml \
    pyrosetta=2022.46+release.f0c6fca=py38_0 \
    python=3.8 \
    python-blosc \
    python-codon-tables \
    python-dateutil \
    python-graphviz \
    pytorch=*=*cuda*cudnn* \
    pytorch-cluster=*=*cu* \
    pytorch-lightning \
    pytorch-mutex=*=*cu* \
    pytorch-scatter=*=*cu* \
    pytorch-sparse=*=*cu* \
    pytorch-spline-conv=*=*cu* \
    pytorch-cuda=11.7 \
    rdkit \
    regex \
    requests \
    rsa \
    ruby \
    scikit-learn \
    scipy \
    seaborn \
    send2trash \
    setuptools \
    simpervisor \
    sympy \
    six \
    statsmodels \
    tensorboard \
    tensorboard-data-server \
    tensorboard-plugin-wit \
    tensorflow \
    tensorflow-estimator \
    deepmodeling::tensorflow-io-gcs-filesystem \
    termcolor \
    tmalign \
    toolz \
    torchaudio=*=*cu* \
    torchvision=*=*cu* \
    tqdm \
    traitlets \
    traittypes \
    typed-ast \
    typing-extensions \
    wandb \
    wheel \
    widgetsnbextension \
    wrapt \
    yt \
    zict \
    omegaconf \
    hydra-core \
    ipdb \
    deepdiff \
    e3nn \
    deepspeed \
    ml-collections \
    assertpy \
    python-dateutil \
    pyrsistent \
    mysql-connector-python \
    pdbfixer \
    cuda-nvcc



pip install ipymol
pip install ml-collections
pip install npose==1.0
pip install pycorn==0.19
pip install pystapler==1.0.3
pip install torch_geometric
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
pip install git+https://github.com/NVIDIA/DeepLearningExamples.git#egg=se3-transformer\&subdirectory=DGLPyTorch/DrugDiscovery/SE3Transformer
pip install git+https://github.com/abroerman/homog.git
pip install dm-haiku
pip install opt_einsum 

# Clean up
micromamba clean -a -y
pip cache purge

%environment 
export PATH=$PATH:/usr/local/cuda/bin
 
%runscript
exec python "$@"
