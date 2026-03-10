Install dependencies
Note

We recommend to use a fresh new conda environment to install verl and its dependencies.

Notice that the inference frameworks often strictly limit your pytorch version and will directly override your installed pytorch if not paying enough attention.

As a countermeasure, it is recommended to install inference frameworks first with the pytorch they needed. For vLLM, if you hope to use your existing pytorch, please follow their official instructions Use an existing PyTorch installation .

First of all, to manage environment, we recommend using conda:

conda create -n verl python==3.12
conda activate verl
Then, execute the install.sh script that we provided in verl:

# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
If you encounter errors in this step, please check the script and manually follow the steps in the script.

[Optional] NVIDIA Apex is recommended for Megatron-LM training, but it’s not needed if you only use FSDP backend. You can install it via the following command, but notice that this steps can take a very long time. It is recommended to set the MAX_JOBS environment variable to accelerate the installation process, but do not set it too large, otherwise the memory will be overloaded and your machines may hang.

# change directory to anywher you like, in verl source code directory is not recommended
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
Install verl
For installing the latest version of verl, the best way is to clone and install it from source. Then you can modify our code to customize your own post-training jobs.

git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
Post-installation
Please make sure that the installed packages are not overridden during the installation of other packages.

The packages worth checking are:

torch and torch series

vLLM
