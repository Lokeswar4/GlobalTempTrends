# GlobalTempTrends

Analyzing global monthly mean temperature trends using GHCN-Monthly Version 4 QCU data. Includes time-series modeling with Random Forest, Gradient Boosting, MLP, and LSTM neural networks.

## Dataset

[GHCN-Monthly Version 4](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly) — global land surface temperature records from NOAA's National Centers for Environmental Information.

## Analysis Pipeline

1. **Data Acquisition** — automated download and parsing of GHCN fixed-width format files
2. **Preprocessing** — station metadata merging, duplicate removal, missing value handling
3. **Outlier Detection** — biweight mean/standard deviation method (robust to non-normal distributions)
4. **Exploratory Analysis** — interactive Folium maps, correlation analysis, trend visualization
5. **Modeling** — 4 models compared with TimeSeriesSplit cross-validation:
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Multi-Layer Perceptron (MLP)
   - LSTM Neural Network (with Keras Tuner hyperparameter search)

## Setup

```bash
pip install -r requirements.txt
cd Global_Temperature_Trends_Analysis_GHCN_V4_QCU
jupyter notebook Global_Temperature_Trends_Analysis_GHCN_V4_QCU.ipynb
```

**Note:** The notebook was originally developed in Google Colab. Some file paths reference `/content/` and may need adjustment for local execution.

## Known Issues

- Hardcoded Google Colab paths (`/content/`) — need to be parameterized for local use
- Scaler is fit on full dataset before train/test split (potential data leakage)
- Some utility functions are defined multiple times in different cells
- Keras imports use deprecated `from keras` instead of `from tensorflow.keras`

## Future Improvements

- Fix data leakage in scaling pipeline (fit scaler only on training data)
- Add seasonal decomposition and autocorrelation analysis
- Include baseline model (seasonal naive) for comparison
- Refactor duplicate functions into reusable modules
- Add random seeds for full reproducibility
- Parameterize file paths with config file

## License

MIT
