# domain-invariant-learning
python 3.9.7
## algo/
implementations of domain invariant learning algo.
--algo_name can switch them.
|file name|note|
|---|---|
|dann_algo.py|DANNs algo https://arxiv.org/pdf/1505.07818|
|coral_algo.py|CoRAL algo https://arxiv.org/abs/1607.01719|
|dan_algo.py|DAN algo https://arxiv.org/abs/1502.02791|
|dann2D_algo.py|see our paper(TODO: Attach in the near future)|
|supervised_algo.py|supervised deep learning boilerplate for comparison test|

## experiments/
implementations of experiment workflow (data load, preprocess, init NN, training, evaluation).
|dir name|data|execution|
|---|---|---|
|make_moons|https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html|`python -m domain-invariant-learning.experiments.make_moons.experiment`|
|ecodataset|https://vs.inf.ethz.ch/res/show.html?what=eco-data|`git clone https://github.com/oh-yu/deep_occupancy_detection/tree/feature/JSAI`<br>`run all cells of 01.ipynb - 05.ipynb`<br>`python -m domain-invariant-learning.experiments.ecodataset_synthetic.experiment`|
|ecodataset_synthetic|see experiment.py logic|`git clone https://github.com/oh-yu/deep_occupancy_detection/tree/feature/JSAI`<br>`run all cells of 01.ipynb - 05.ipynb`<br>`python -m domain-invariant-learning.experiments.ecodataset_synthetic.experiment`|
|HHAR|https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition|`download data`<br>`python -m domain-invariant-learning.experiments.HHAR.experiment`|
|MNIST|https://github.com/mashaan14/MNIST-M/tree/main|`download data`<br>`python -m domain-invariant-learning.experiments.MNIST.experiment`|

## networks/
implementations of networks which include layers, fit method, predict method, predict_proba method.
Domain Invariant Laerning and Without Adapt and Train on Target related free params should be set here.
|file name|note|
|---|---|
|dann.py|Figure 4: from https://arxiv.org/pdf/1505.07818|
|codats.py|Figure 3: from https://arxiv.org/pdf/2005.10996|
|danns_2d.py|see our paper(TODO: Attach in the near future)|
|isih-DA.py|see our paper(TODO: Attach in the near future)|

## utils/
Definition of generic functions to be called in multiple locations within the above dir structure.



  







