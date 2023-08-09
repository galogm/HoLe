# CIKM'23 - HoLe: Homophily-enhanced Structure Learning for Graph Clustering

## Installation

- See [requirements.txt](./requirements.txt) and [requirements-dev.txt](./requirements-dev.txt) for requirements.
- See [install-dev.sh](./.ci/install-dev.sh) and [install.sh](./install.sh) for installation scripts.

```bash
# Install python>=3.7.16.

# Install dev requirements:
$ bash .ci/install-dev.sh

# Install requirements.
# Cuda defaults to 11.3. Change it if necessary.
$ bash .ci/install.sh
```

## Usage

```bash
# Activate the env, e.g., for linux run:
$ source .venv/bin/activate

# Run HoLe:
$ python main.py --dataset Cora --gpu_id 0
```

## Citation

```bib
{

}
```
