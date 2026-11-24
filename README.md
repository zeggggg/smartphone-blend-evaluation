# Does tuning a prediction blend improve its score on new data?

In Kaggle's *Predicting Smartphone Addiction* competition, my final blend improved public leaderboard AUC by **0.00002**. Both versions scored **0.97103** on the private leaderboard at the reported precision. That left me with a question: how much of a small gain can I get simply by trying more weights on the same rows? [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e8)

## A leaderboard I could test

I started with two sets of out-of-fold predictions. Each row had predictions from models trained on other folds. In each of 40 repetitions, I used 59,260 rows as a mock public leaderboard, tried 21 blend weights, and selected the weight with the highest AUC. I then scored that choice on the remaining 632,109 rows.

| Where the blend was scored | Average AUC gain recorded in the competition notebook |
|---|---:|
| Rows used to choose the weight | +0.000003 |
| Remaining rows | +0.000000 |

The averages are rounded to six decimals. In this experiment, the small gain visible during selection disappeared at that precision on fresh rows. It changed how I approach a new prediction problem: I now want to understand what each model contributes before I start combining them.

## Run the evaluation

The public script demonstrates the experiment using synthetic data. It also accepts your own predictions. Its reference blend is fixed at 50/50, and it breaks exact ties by preferring a weight closest to 50/50. The competition figures above come from my original notebook. Reproducing those exact scores requires the original prediction vectors.

```sh
python3 -m pip install numpy scikit-learn
python3 evaluate.py --demo
```

Read the two mean gains together. The selection gain measures how much tuning helps on the rows used to pick a weight. The held-out gain measures how much of that advantage carries over. The final count tells you how often the chosen weight loses to the fixed blend on held-out rows. Repeated splits share observations.

For your own data, supply a CSV with columns `label,pred_a,pred_b`. Use one row per observation, binary labels, and two aligned out-of-fold prediction scores on comparable scales. The following command uses the original selection-set size. [Evaluation script](evaluate.py)

```sh
python3 evaluate.py predictions.csv --selection-size 59260 --repeats 40 --seed 42
```

## My contribution

I compared blends built from predictions shared by the Kaggle community and examined what changed between selection and evaluation. AI tools assisted with coding, analysis, and preparing this public write-up.
