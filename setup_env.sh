source ~/anaconda3/etc/profile.d/conda.sh

ENV_NAME=dash

conda deactivate
conda env list
conda env remove -n ${ENV_NAME}
conda env list

conda create -n ${ENV_NAME} python -y
conda activate ${ENV_NAME}
conda activate ${ENV_NAME}
conda env list

# pip cache purge
pip install --no-cache-dir dash
pip install --no-cache-dir pandas