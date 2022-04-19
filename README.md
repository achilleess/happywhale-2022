# Kaggle Happywhale 2022 Competition, 21st place code

This repository contains a part of the code for the Happywhale 2022 competition. But it lucks the precise instructions to fully reproduce the results.

#### Enironment
```
conda env create -f train_environment.yml
conda activate happywhale
```

#### Solution
https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/319828

#### Train
to train and test cnn models
```
python3 tools/train.py <fold_num>
python3 tools/test_model.py <fold_num>
```

to train and test MLP models
```
python3 tools/train_mlp.py <fold_num>
python3 tools/test_mlp.py <fold_num>
```
