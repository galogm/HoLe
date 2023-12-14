# CIKM'23 - HoLe: Homophily-enhanced Structure Learning for Graph Clustering

An implementation of the CIKM'23 paper: [Homophily-enhanced Structure Learning for Graph Clustering](https://dl.acm.org/doi/pdf/10.1145/3583780.3614915).

## Installation

- See [requirements.txt](./requirements.txt) and [requirements-dev.txt](./requirements-dev.txt) for requirements.
- See [install-dev.sh](./.ci/install-dev.sh) and [install.sh](./install.sh) for installation scripts.

```bash
# Install python>=3.8.

# Install dev requirements:
$ bash .ci/install-dev.sh

# Install requirements.
# Cuda defaults to 11.3. Change it if necessary.
$ bash .ci/install.sh
```

## Usage

```bash
# Activate the env, e.g., for linux run:
$ source .env/bin/activate

# Run HoLe:
$ python main.py --dataset Cora --gpu_id 0
```

## Citation
Please cite our paper if you use this code in your own work:
```bib
@inproceedings{gu2023homophily,
  title={Homophily-enhanced Structure Learning for Graph Clustering},
  author={Gu, Ming and Yang, Gaoming and Zhou, Sheng and Ma, Ning and Chen, Jiawei and Tan, Qiaoyu and Liu, Meihan and Bu, Jiajun},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={577--586},
  year={2023}
}
```
