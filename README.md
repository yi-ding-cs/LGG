# LGGNet
This is the PyTorch implementation of the LGG using [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset in our paper:

Yi Ding, Neethu Robinson, Qiuhao Zeng, Cuntai Guan, "LGGNet: Learning from Local-Global-Graph Representations for Brain-Computer Interface", under review of _**IEEE Transactions on Neural Networks and Learning Systems(TNNLS)**_, preprint available at [arXiv](https://arxiv.org/abs/2105.02786)

It is a neurologically inspired graph neural network to learn local-global-graph representations from Electroencephalography (EEG) for a Brain-Computer Interface (BCI).
# Network structure of LGGNet
<p align="center">
<img src="https://user-images.githubusercontent.com/83038743/159688807-458dabff-fd89-400a-bad4-e0510fd90671.png" width=800 align=center>
</p>

<p align="center">
 Fig.1 LGGNet structure
</p>

LGGNet has three main functional blocks, the temporal learning block, the graph building block, and the graph learning block. The temporal convolutional layer is shown in the temporal learning block (a). Building the local-global graphs of EEG using the attentively fused temporal representations is illustrated in the graph building block (b). The local and global graph-filtering layers are shown in the graph learning block (c). The temporal convolutional layer aims to learn dynamic temporal representations from EEG directly instead of human extracted features. The kernel-level attentive fusion layer will fuse the information learned by different temporal kernels to increase the learning capacity of LGGNet. The local graph-filtering layer learns the brain activities within each local region. Then the global graph-filtering layer with a trainable adjacency matrix will be applied to learn complex relations among different local regions. Four local graphs are shown in the figure for illustration purposes only. To know more about the detailed local-global-graph definitions please refer to our paper.
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

# Acknowledgment
The author would like to thank Chengxuan Tong for checking the code

# Cite
Please cite our paper if you use our code in your own work:

```
@misc{ding2021lggnet,
      title={LGGNet: Learning from Local-Global-Graph Representations for Brain-Computer Interface}, 
      author={Yi Ding and Neethu Robinson and Qiuhao Zeng and Cuntai Guan},
      year={2021},
      eprint={2105.02786},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```
