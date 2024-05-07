pip install --user virtualenv
mkdir grad-attack-env
virtualenv grad-attack-env
source grad-attack-env/bin/activate
python3 -m pip install -r requirements.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pillow==9.5.0
pip3 install pytorch_lightning==1.3.0
pip3 install lpips
pip install --upgrade jax==0.3.15 jaxlib==0.3.15 -f https://storage.googleapis.com/jax-releases/jax_releases.html
deactivate