# Shipments Without Downtime - Solution (Solo Track)

*[Русская версия](README.ru.md)*

My solution for the [WB Wildhack 2026: "Shipments Without Downtime"](https://wbspace.wb.ru/competitions/otgruzki-bez-prostoev) hackathon.

## Task

The goal is to predict the shipment volume (`target_1h`) for every route (`route_id`) 4 hours ahead. The input is historical time series: the target itself and six service statuses (`status_1`...`status_6`). The series have a step of 30 minutes (48 points per day).

**Metric:** `WAPE + |Relative Bias|`

```
WAPE = Σ|y_pred - y_true| / Σ|y_true|
RBias = |Σy_pred / Σy_true - 1|
```

## Data

- `train_solo_track.parquet` - history for each route: `timestamp`, `route_id`, `status_1..status_6`, `target_1h`.
- `test_solo_track.parquet` - test set where `target_1h` has to be predicted.

The notebook loads the files from Google Drive and is written to run in Google Colab.
The data itself is not published here because of the hackathon rules.


## Approaches I Tried

The first model I ran was CatBoost with lag features, but lags alone were clearly not enough. Then I started to add different aggregations (rolling mean and std, statistics by hour and day of week, the "speed" of status changes). This gave only a small gain, and the score stopped again at about 0.355 on the leaderboard.

After that I tried different models from `darts`. I also tried transformers, but they were too slow to train and still worse than CatBoost on validation.

Then I found **TiDE (Time-series Dense Encoder)** - an architecture built on dense layers. It trains very fast and works as well as transformers. It gave me a better score, so until the end of the hackathon I was tuning its hyperparameters (encoder/decoder depth, hidden size, history length, loss function).

In the end the model uses:
- `status_1...status_6` as past covariates;
- cyclic features `hour` and `dayofweek` as future covariates;
- relative position in time (past/future encoders from `darts`);
- scaling of the input and the target with `Scaler`.

I also applied a bias correction: on validation I computed the coefficient `bias_coef = Σy_true / Σy_pred` and multiplied the final predictions by it. This pushed the Relative Bias to zero separately, without retraining the model (I used it both for CatBoost and for TiDE).

## Result

The best local validation score was reached by **Darts TiDE** with an honest backtest (horizon of 8 steps, stride 8):

```
WAPE + Bias ≈ 0.329
```
