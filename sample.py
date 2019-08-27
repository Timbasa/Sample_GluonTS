from functools import partial

import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.trainer import Trainer

epochs = 5
num_batches_per_epoch = 10
dataset_name = "m4_hourly"
dataset = get_dataset(dataset_name)
results = []
# If you want to use GPU, please set ctx="gpu(0)"
estimators = [
    # partial(
    #     DeepAREstimator,
    #     trainer=Trainer(
    #         ctx="cpu",
    #         epochs=epochs,
    #         num_batches_per_epoch=num_batches_per_epoch
    #     )
    # ),
    partial(
        MQCNNEstimator,
        trainer=Trainer(
            ctx="cpu",
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch
        )
    ),
]

for estimator in estimators:
    estimator = estimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq
    )
    predictor = estimator.train(dataset.train)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_eval_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    results.append(eval_dict)

df = pd.DataFrame(results)
sub_df = df[
    [
        "dataset",
        "estimator",
        "mean_wQuantileLoss",
    ]
]
print(sub_df)
