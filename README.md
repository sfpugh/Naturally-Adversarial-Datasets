# Curating Naturally Adversarial Datasets for Learning-Enabled Medical Cyber-Physical Systems

This repository implements an approach to curating naturally adversarial datasets via adversarial ordering presented in this [paper](https://browse.arxiv.org/abs/2309.00543) accepted to ICCPS 2024.

## Instructions

Clone this repository.

```
git clone https://github.com/sfpugh/Naturally-Adversarial-Datasets
cd Naturally-Adversarial-Datasets
```

The paper results were generated on a Linux machine with Ubuntu 20.04 and Python 3.8. We provide a Dockerfile to construct an image to this specification. To build the docker image, use the following command:

```
docker build -t nad .
```

To create and run a container from this image `nad`, use the following command:

```
docker run -it nad
```

Decompress the data:

```
cd data
tar -xzvf data.tar.gz
cd ..
```

To apply the approach, run from `main.py` with appropriate arguments. All arguments are listed in `main.py`. For example, to run the approach on the `spo2_hr_lo` dataset using probabilistic labeler majority vote, you can use the following command:

```
python main.py --dataset spo2_hr_lo --default_pred 1 --pl majorityvote --evaluate
```

Finally, to reproduce Table III from the [paper](https://browse.arxiv.org/abs/2309.00543) run `generate_results.sh` in the `scripts/` directory.

```
cd scripts
bash generate_results.sh
```

## Citation

Please cite this paper in your publications if this code helps your research.

```
@misc{pugh2023curating,
      title={Curating Naturally Adversarial Datasets for Learning-Enabled Medical Cyber-Physical Systems}, 
      author={Sydney Pugh and Ivan Ruchkin and Insup Lee and James Weimer},
      year={2023},
      eprint={2309.00543},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
