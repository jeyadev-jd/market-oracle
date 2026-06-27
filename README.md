# Market Oracle

Commodity price prediction pipeline built for the Mitsui Commodity Prediction Challenge. Combines time-series forecasting (NeuralProphet, Chronos) with deep learning (PyTorch) to predict target commodity pairs.

## Contents

- [marketoracle.ipynb](marketoracle.ipynb) — main notebook: data loading, feature engineering, model training, and prediction pipeline.

## Stack

- PyTorch
- NeuralProphet
- Chronos (chronos_forecasting)
- Hugging Face Transformers
- pandas / numpy / scikit-learn / scipy

## Usage

Designed to run on Kaggle with offline package wheels and the Mitsui Commodity Prediction Challenge dataset attached. Adjust the `WHEELS_DIR`, `TRAIN_CSV`, `LABELS_CSV`, and `PAIRS_CSV` paths at the top of the notebook to match your environment.

## License

MIT — see [LICENSE](LICENSE).
