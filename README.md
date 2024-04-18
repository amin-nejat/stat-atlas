# Statistical Atlas of C. elegans Neurons
![Statistical Atlas of C. elegans Neurons](https://github.com/amin-nejat/stat-atlas/assets/5959554/9d8ec6c4-dcde-4d92-b2b1-43e7a1536973)

Constructing a statistical atlas of neuron positions in the nematode Caenorhabditis elegans enables a wide range of applications that require neural identity. These applications include annotating gene expression, extracting calcium activity, and evaluating nervous system mutations. Large complete sets of neural annotations are necessary to determine canonical neuron positions and their associated confidence regions. Recently, a transgene of C. elegans (“NeuroPAL”) has been introduced to assign correct identities to all neurons in the worm via a deterministic, fluorescent colormap. This strain has enabled efficient and accurate annotation of worm neurons. Using a dataset of 10 worms, we propose a statistical model that captures the latent means and covariances of neuron locations, with efficient optimization strategies to infer model parameters. We demonstrate the utility of this model in two critical applications. First, we use our trained atlas to automatically annotate neuron identities in C. elegans at the state-of-the-art rate. Second, we use our atlas to compute correlations between neuron positions, thereby determining covariance in neuron placement.

See **[our paper](https://link.springer.com/chapter/10.1007/978-3-030-59722-1_12)** for further details:


```
@inproceedings{varol2020statistical,
  title={Statistical atlas of C. elegans neurons},
  author={Varol, Erdem and Nejatbakhsh, Amin and Sun, Ruoxi and Mena, Gonzalo and Yemini, Eviatar and Hobert, Oliver and Paninski, Liam},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2020: 23rd International Conference, Lima, Peru, October 4--8, 2020, Proceedings, Part V 23},
  pages={119--129},
  year={2020},
  organization={Springer}
}
```
**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (anejatbakhsh@flatironinstitute.org) if you have questions. If you are using this code package, please cite our paper.

## A short and preliminary guide

### Installation Instructions

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it

```
conda create -n statatlas python=3.8
conda activate statatlas
```

3. Clone the repository

```
https://github.com/amin-nejat/stat-atlas.git
cd stat-atlas
```

4. Install package requirements

```
pip install -r requirements.txt
```

5. Run either using the demo notebook file in the notebooks folder.


Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.
