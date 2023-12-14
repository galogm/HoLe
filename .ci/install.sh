# install python 3.8, e.g., if using pyenv, run:
# pyenv install 3.8
# pyenv local 3.8

# create and activate virtual environment
if [ ! -d '.env' ]; then
    python3 -m venv .env && echo create venv
else
    echo venv exists
fi

source .env/bin/activate

# # update pip
# python -m pip install -U pip

# torch cuda 11.3
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# dgl cuda 11.3
python -m pip install dgl==1.1 -f https://data.dgl.ai/wheels/cu113/repo.html
python -m pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

# install requirements
python -m pip install -r requirements.txt

echo install requirements successfully!
