# install python 3.8, e.g., if using pyenv, run:
# pyenv install 3.8
# pyenv local 3.8

# create and activate virtual environment
python3 -m venv .env
source .env/bin/activate

# update pip
# add sources if needed: ` -i https://pypi.tuna.tsinghua.edu.cn/simple/`
python -m pip install -U pip
# install requirements
python -m pip install -r requirements-dev.txt

# pre-commit install

echo install DEV requirements successfully!
