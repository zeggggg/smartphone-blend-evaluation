# Does a better leaderboard blend hold up on fresh rows?

During Kaggle's Predicting Smartphone Addiction competition, I compared blends of community model predictions. My final blend gained 0.00002 AUC on the public leaderboard. Both versions scored 0.97103 on the private leaderboard at the displayed precision. That small gap made me want to understand how much of an apparent gain I could get just by trying different weights. [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e8)

## What I measured

My competition notebook describes a check using two out-of-fold prediction vectors. Each entry is a prediction for a training row from a model fitted on other folds. In each of 40 repetitions, a subset of 59,260 rows served as a pretend public leaderboard. I selected a weight from 21 candidates on those rows, then evaluated the choice on the remaining 632,109 rows. The notebook recorded an average apparent gain of 0.000003 AUC on the selection rows and 0.000000 on the remaining rows, rounded to six decimals. I used AI assistance for coding and analysis and relied on predictions contributed by the competition community.

## Try the evaluation

This repository includes a standalone implementation of the split-and-select exercise. It compares the selected blend to a fixed 50/50 reference on both subsets. The reference, random seed and tie rule are explicit for this implementation. The historical figures above are transcribed from the competition notebook. The competition prediction vectors are required to rerun the historical experiment.

With NumPy and scikit-learn installed, run the synthetic example:

```sh
python3 evaluate.py --demo
```

To use your own aligned, out-of-fold predictions, supply a CSV with one row per observation and the columns `label,pred_a,pred_b`. Labels must be 0 or 1. Predictions should be on comparable scales. For the original split size:

```sh
python3 evaluate.py predictions.csv --selection-size 59260 --repeats 40 --seed 42
```

The script prints average AUC gains over the reference and counts how often the tuned blend does worse on held-out rows. Positive selection gain is expected because the reference weight is among the candidates. The held-out gain is the quantity I would look at before choosing the more elaborate blend. Repeated splits reuse observations, so the repetitions are not independent datasets.
