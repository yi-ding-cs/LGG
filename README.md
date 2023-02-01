# LGGNet
This is the PyTorch implementation of the LGGNet using [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset in our paper:

Yi Ding, Neethu Robinson, Chengxuan Tong, Qiuhao Zeng, Cuntai Guan, "LGGNet: Learning from Local-Global-Graph Representations for Brain-Computer Interface", accepted as a regular paper in the _**IEEE Transactions on Neural Networks and Learning Systems(TNNLS)**_, available at [IEEE Xplore](https://ieeexplore.ieee.org/document/10025569)

It is a neurologically inspired graph neural network to learn local-global-graph representations from Electroencephalography (EEG) for a Brain-Computer Interface (BCI).
# Network structure of LGGNet
<p align="center">
<img src="https://user-images.githubusercontent.com/83038743/205667640-e3784e1b-4441-4c51-b269-3ce0417309b2.png" width=900 align=center>
</p>

<p align="center">
 Fig.1 LGGNet structure
</p>

LGGNet has two main functional blocks: the temporal learning block and the graph learning block. The temporal convolutional layer and the kernel-level attentive fusion layer are shown in the temporal learning block (A). The local and global graph-filtering layers are shown in the graph learning block (B). The temporal convolutional layer aims to learn dynamic temporal representations from EEG directly instead of human-extracted features. The kernel-level attentive fusion layer will fuse the information learned by different temporal kernels to increase the learning capacity of LGGNet. The local graph-filtering layer learns the brain activities within each local region. Then the global graph-filtering layer with a trainable adjacency matrix will be applied to learn complex relations among different local regions. Four local graphs are shown in the figure for illustration purposes only. To know more about the detailed local-global-graph definitions please refer to our paper.

# Graph definitions for LGGNet
<p align="center">
<img src="https://user-images.githubusercontent.com/83038743/215672010-ccfcf760-6937-4754-8bb8-6da4d83e2b16.png" width=800 align=center>
</p>

<p align="center">
 Fig.2 Three types of local-global-graph (LGG) definitions.
</p>

Fig.2 shows three types of LGG definitions. (a) General LGG definition. This local-graph structure is defined according to the 10–20 system. Each local graph reflects the brain activities of a certain brain functional area. (b) Frontal LGG definition. Based on the general LGG, the neuroscience evidence of frontal asymmetry patterns in frontal areas is further considered. Six frontal local graphs that are symmetrically located on the left and right frontal areas of the brain are added to learn more discriminative information. (c) Hemisphere LGG definition. The symmetrical local graphs are added for all the functional areas defined in the general LGG. The nodes in a local graph are in the same color. The dotted lines are the local graphs. This diagram illustrates the definition for the 62 channel EEG.

# Prepare the python virtual environment
Please create an anaconda virtual environment by:

> $ conda create --name LGG python=3.8

Activate the virtual environment by:

> $ conda activate LGG

Install the requirements by:

> $ pip3 install -r requirements.txt

# Run the code
Please download the DEAP dataset at [this website](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). Please place the "data_preprocessed_python" folder in the same location as the script. To run the code for the emotion (valence) classification, please type the following command in the terminal:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'V' --graph-type 'gen'

To run the code for preference (liking) classification, please type the following command in the terminal:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'L' --graph-type 'hem'

The results will be saved into "result_DEAP.txt" located at './save/result/'. 

# Reproduce the results
We highly suggest running the code on a Ubuntu 18.04 or above machine using anaconda with the provided requirements to reproduce the results. 
You can also download the saved model at [this website](https://drive.google.com/file/d/12lIbX6ti7cDCv3mVDY7TTd4QIc2cNEYE/view?usp=sharing) to reproduce the results in the paper. After extracting the downloaded "save.zip", please place it at the same location as the scripts, and run the code by:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'V' --graph-type 'gen' --reproduce

# Apply LGGNet to other datasets
If you are interested to apply LGGNet to other datasets, you can follow the below example. 

## Example of the usage
```python
from networks import LGGNet

original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz',
                  'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',
                  'PO4', 'O2']

# Define proper channel orders for the local-global graphs in LGGNet. Please refer to three graph definitions (general, frontal, hemesphere).
graph_general_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                     ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                     ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                     ['T7'], ['T8']]

graph_idx = graph_general_DEAP   # The general graph definition for DEAP is used as an example.
idx = []
num_chan_local_graph = []
for i in range(len(graph_idx)):
    num_chan_local_graph.append(len(graph_idx[i]))
    for chan in graph_idx[i]:
        idx.append(original_order.index(chan))

data = torch.randn(1, 1, 32, 512)  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
data = data[:, :, idx, :]  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)

LGG = LGGNet(
    num_classes=2,
    input_size=(1, 32, 512),
    sampling_rate=128,
    num_T=64,  # num_T controls the number of temporal filters
    out_graph=32,
    pool=16,
    pool_step_rate=0.25,
    idx_graph=num_chan_local_graph,
    dropout_rate=0.5
)

preds = LGG(data)
```

# CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |

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
