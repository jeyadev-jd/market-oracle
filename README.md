# Market Oracle

Commodity price prediction pipeline built for the Mitsui Commodity Prediction Challenge. Combines time-series forecasting (NeuralProphet, Chronos) with deep learning (PyTorch) to predict target commodity pairs.

## Contents

- [marketoracle.ipynb](marketoracle.ipynb) — main notebook: data loading, feature engineering, model training, and prediction pipeline.
- [rf_baseline.ipynb](rf_baseline.ipynb) — RandomForest baseline with technical indicators (SMA/EMA/RSI/MACD/Bollinger Bands).

## Architecture (marketoracle.ipynb)

Ensemble of 9 models combined via a Bayesian meta-ensemble:

- **Chronos** (`amazon/chronos-t5-large`) — foundation time-series forecaster
- **NeuralProphet** — seasonality/trend forecaster
- **RetrievalAugmentedDiffusion (RAD)** — pattern DB + encoder, similarity-weighted retrieval
- **NeuralCDE** — neural controlled differential equation w/ attention readout
- **TimeMoESystem** — 32-expert mixture-of-experts, top-4 routing
- **MambaStateSpaceModelFixed** — stacked-LSTM fallback for state-space modeling
- **TemporalGraphAttentionNetworkFixed** — cross-asset correlation graph signal
- **CausalInferenceEngine** — Granger causality graph (networkx)
- **AdvancedMarketRegimeDetector** — classifies bull/bear/stable/volatile/neutral regimes

`CompleteUltimatePredictor` wires all models together; `EnhancedBayesianEnsemble` weights each model's output by its rolling error history and produces a mean prediction + 95% confidence interval. Final predictions written to `submission.parquet`.

## Stack

- PyTorch
- NeuralProphet
- Chronos (chronos_forecasting)
- Hugging Face Transformers
- networkx
- pandas / numpy / scikit-learn / scipy

## Usage

Designed to run on Kaggle with offline package wheels and the Mitsui Commodity Prediction Challenge dataset attached. Adjust the `WHEELS_DIR`, `TRAIN_CSV`, `LABELS_CSV`, and `PAIRS_CSV` paths at the top of the notebook to match your environment.

## License

MIT — see [LICENSE](LICENSE).
