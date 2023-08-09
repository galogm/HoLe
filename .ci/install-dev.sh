# install python 3.7.16, e.g., if using pyenv, run:
# pyenv install 3.7.16
# pyenv local 3.7.16

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# update pip
# add sources if needed: ` -i https://pypi.tuna.tsinghua.edu.cn/simple/`
python -m pip install -U pip
# install requirements
python -m pip install -r requirements-dev.txt

# pre-commit install

echo install DEV requirements successfully!
