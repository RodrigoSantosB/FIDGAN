# FID-GAN

This repository contains the code for the paper "Intrusion Detection for Cyber-Physical Systems using Generative Adversarial Networks in Fog Environment," published in the IEEE Internet of Things Journal. The paper was authored by Paulo Freitas de Araujo-Filho, Georges Kaddoum, Divanilson R. Campelo, Aline Gondim Santos, David Macêdo, and Cleber Zanchettin.

Link to the paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/9199878)

For more information about the code and the paper, contact us at: **paulo.freitas-de-araujo-filho.1@ens.etsmtl.ca**

If you wish to cite our work, use:

> **P. F. de Araujo-Filho, G. Kaddoum, D. R. Campelo, A. G. Santos, D. Macêdo, and C. Zanchettin, "Intrusion Detection for Cyber-Physical Systems using Generative Adversarial Networks in Fog Environment," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2020.3024800.**

## Requirements

Before running the code, ensure you have the following:

- **Python 3.7**
- **TensorFlow 2.1** or later
- Additional libraries listed in the `environment.yaml` file

To install dependencies, run:
```sh
conda create -f environment.yaml
conda activate fid-gan
```

## Setup and Execution

The experiments use three datasets. Please refer to the paper for instructions on acquiring them.

### Train the GAN model
```sh
python scripts/RGAN.py --settings_file <nslkdd/wadi/swat>
```

### Train the Autoencoder model
```sh
python scripts/autoencoder.py --settings_file <nslkdd/wadi/swat>
```

### Perform Anomaly Detection
```sh
python scripts/AD_autoencoder.py --settings_file <nslkdd_test/wadi_test/swat_test>

python scripts/AD_computeResults.py --settings_file <nslkdd_test/wadi_test/swat_test>
```

## Installing and Running Shell Scripts (`.sh`)

If the repository includes `.sh` scripts, follow these steps to install and execute them:

1. **Grant execution permission to the script:**
```sh
chmod +x script.sh
```

2. **Run the script:**
```sh
./script.sh
```

If the script needs to be run as a superuser:
```sh
sudo ./script.sh
```

## Additional Reference

Part of this repository contains code from the paper:
> **D. Li, D. Chen, B. Jin, L. Shi, J. Goh, and S.-K. Ng, "MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks," in Springer Int. Conf. on Artif. Neural Netw., 2019, pp. 703–716.**

