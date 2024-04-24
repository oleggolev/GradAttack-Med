pip install --user virtualenv
mkdir grad-attack-env
virtualenv grad-attack-env
source grad-attack-env/bin/activate
python3 -m pip install -r requirements.txt
pip3 install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
deactivate