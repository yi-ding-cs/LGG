# LGGNet
This is the PyTorch implementation of the LGG using [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset in our paper:

Yi Ding, Neethu Robinson, Chengxuan Tong, Qiuhao Zeng, Cuntai Guan, "LGGNet: Learning from Local-Global-Graph Representations for Brain-Computer Interface", accepted as a regular paper in the _**IEEE Transactions on Neural Networks and Learning Systems(TNNLS)**_, available at [IEEE Xplore](https://ieeexplore.ieee.org/document/10025569)

It is a neurologically inspired graph neural network to learn local-global-graph representations from Electroencephalography (EEG) for a Brain-Computer Interface (BCI).
# Network structure of LGGNet
<p align="center">
<img src="https://user-images.githubusercontent.com/83038743/205667640-e3784e1b-4441-4c51-b269-3ce0417309b2.png" width=800 align=center>
</p>

<p align="center">
 Fig.1 LGGNet structure
</p>

LGGNet has two main functional blocks: the temporal learning block and the graph learning block. The temporal convolutional layer and the kernel-level attentive fusion layer are shown in the temporal learning block (A). The local and global graph-filtering layers are shown in the graph learning block (B). The temporal convolutional layer aims to learn dynamic temporal representations from EEG directly instead of human extracted features. The kernel-level attentive fusion layer will fuse the information learned by different temporal kernels to increase the learning capacity of LGGNet. The local graph-filtering layer learns the brain activities within each local region. Then the global graph-filtering layer with a trainable adjacency matrix will be applied to learn complex relations among different local regions. Four local graphs are shown in the figure for illustration purposes only. To know more about the detailed local-global-graph definitions please refer to our paper.
# Prepare the python virtual environment
Please create an anaconda virtual environment by:

> $ conda create --name LGG python=3.8

Activate the virtual environment by:

> $ conda activate LGG

Install the requirements by:

> $ pip3 install -r requirements.txt

# Run the code
Please download the DEAP dataset at [this website](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). Please place the "data_preprocessed_python" folder at the same location of the script. To run the code for emotion (valence) classification, please type the following command in terminal:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'V' --graph-type 'gen'

To run the code for preference (liking) classification, please type the following command in terminal:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'L' --graph-type 'hem'

The results will be saved into "result_DEAP.txt" located at './save/result/'. 

# Reproduce the results
We highly suggest to run the code on a Ubuntu 18.04 or above machine using anaconda with the provided requirements to reproduce the results. 
You can also download the saved model at [this website](https://drive.google.com/file/d/12lIbX6ti7cDCv3mVDY7TTd4QIc2cNEYE/view?usp=sharing) to reproduce the results in the paper. After extracting the downloaded "save.zip", please place it at the same location of the scripts, run the code by:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'V' --graph-type 'gen' --reproduce True

# Cite
Please cite our paper if you use our code in your own work:

```
@ARTICLE{10025569,
  author={Ding, Yi and Robinson, Neethu and Tong, Chengxuan and Zeng, Qiuhao and Guan, Cuntai},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={LGGNet: Learning From Local-Global-Graph Representations for Brain–Computer Interface}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2023.3236635}}

```
