# Evaluating Gaze Event Detection Algorithms: Impacts on Machine Learning-based Classification and Psycholinguistic Statistical Modeling
![overview](https://github.com/user-attachments/assets/3411ad03-f2f1-47c3-af26-e20341b713b3)
## HOWTO
### ML
to reproduce the classification results using machine learning, run the following command for each dataset and detection method.

```bash
python evaluate_classification_event_detection.py --dataset DATASET --detection-method DETECTION_METHOD --flag-redo
```

Note: this mgiht take several weeks, depending on your setup.


you can plot the results, using the notebooks provided.


### PLA

for pla, clone the two following repositories:

```bash
git clone git@github.com:dili-lab/EMTeC
git clone git@github.com:dili-lab/potec
```

and follow their steps to download and merge the dataframes.

for `EMTeC` we used the following code snippet to download the data using pymovements:

```python
import pymovements as pm


pm.Dataset('EMTeC', 'data/EMTeC').download()
```

afterwards, run the following python scripts for each dataset and detection method:

```bash
python run_pla_preprocess.py --dataset DATASET --detection-method DETECTION_METHOD --flag-redo
```

Note: this might take several days, depending on your setup.

then, you need to merge the dataframes, using

```bash
python merge_rm_dfs.py --dataset DATASET --detection-method DETECTION_METHOD
```

next, you have to fit the hierarchical models, using `baseline_analyses.R`.

you can use the bash script `run_regression_models.sh`.

depending on your setup, this might take several days/weeks.

lastly, to extract the plots, use `extract_and_plot.R`.


NOTE: PLA depends on a rebased pymovements branch `WIP-entire-aoi-row-instead-ia`.
