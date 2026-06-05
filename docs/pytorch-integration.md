# PyTorch / TorchUncertainty integration

For **deep learning** workflows (Lightning training loops, vision classifiers), DEUP is
available as a post-processing method in
[TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty).

## When to use which package

| Need | Package |
|---|---|
| sklearn / LightGBM / tabular | **`deup`** (this repo) |
| Time-series / finance panels | **`deup`** — `CrossSectionalDEUP`, purged walk-forward |
| PyTorch / Lightning / vision | **TorchUncertainty** `DEUP` post-processor |
| Both | Install both; cite Lahlou et al. (2023) for the method |

## TorchUncertainty usage

```python
from torch_uncertainty.post_processing import DEUP

deup = DEUP(task="classification", model=trained_model, n_folds=5)
deup.fit(calibration_dataloader)
uncertainty = deup(batch)  # g(x) >= 0
```

With OOD evaluation in `ClassificationRoutine`:

```python
from torch_uncertainty.routines import ClassificationRoutine

baseline = ClassificationRoutine(
    num_classes=10,
    model=model,
    post_processing=deup,
    ood_criterion="deup",
    eval_ood=True,
)
```

Tutorial: [TorchUncertainty DEUP tutorial](https://github.com/torch-uncertainty/torch-uncertainty/blob/main/auto_tutorial_source/Post_Hoc_Methods/tutorial_deup.py)

Upstream PR: [torch-uncertainty#313](https://github.com/torch-uncertainty/torch-uncertainty/pull/313)

## Method credit

DEUP is due to Lahlou et al. (2023, TMLR). TorchUncertainty hosts the PyTorch
post-processing integration; this package remains the home for sklearn-compatible and
time-series-correct DEUP.
