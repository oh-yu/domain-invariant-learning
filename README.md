# domain-invariant-learning
python 3.9.7  
![dann](/make_moons_experiemnt_dann.png) ![without_adapt](/make_moons_experiment_withoutadapt.png)

## algo/
implementations of domain invariant learning algo.
|file name|note|
|---|---|
|algo.py|DANN algo https://arxiv.org/pdf/1505.07818|
|coral_alog.py|CoRAL algo https://arxiv.org/abs/1607.01719
|dan_alog.py|DAN algo https://arxiv.org/abs/1502.02791|
## experiments/
implementations of experiment workflow (data load, preprocess, init NN, training, evaluation).
|dir name|data|execution|
|---|---|---|
|make_moons|https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html|python -m domain-invariant-learning.experiments.make_moons.experiment|
|ecodataset|https://vs.inf.ethz.ch/res/show.html?what=eco-data|
git clone https://github.com/oh-yu/deep_occupancy_detection/tree/feature/JSAI  
run all cells of 01.ipynb - 05.ipynb  
python -m domain-invariant-learning.experiments.ecodataset_synthetic.experiment  
|
|ecodataset_synthetic|see experiment.py logic|
git clone https://github.com/oh-yu/deep_occupancy_detection/tree/feature/JSAI  
run all cells of 01.ipynb - 05.ipynb   
python -m domain-invariant-learning.experiments.ecodataset_synthetic.experiment  
|

## networks/
implementations of networks which include layers, fit method, predict method, predict_proba method.
Domain Invariant Laerning and Without Adapt and Train on Target related free params should be set here.

## utils/
Definition of generic functions to be called in multiple locations within the above dir structure.



  







