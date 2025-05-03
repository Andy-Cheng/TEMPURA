# Install conda environment
conda create -n tempura python=3.12.9 -y
conda activate tempura

# Install dependencies
pip install torch==2.5.1 torchvision==0.20.1  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git accelerate
pip install flash-attn==2.5.8 --no-build-isolation

# Login to your wandb account
wandb login $WANDB_API_KEY
# Login to your huggingface account
huggingface-cli login --token $HF_TOKEN

# Install TEMPURA
cd TEMPURA
pip install -e .



