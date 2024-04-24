pip install --user virtualenv
mkdir grad-attack-env
virtualenv grad-attack-env
source grad-attack-env/bin/activate
python3 -m pip install -r requirements.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
deactivate