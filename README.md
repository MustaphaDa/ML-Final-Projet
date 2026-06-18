# Airbnb Price Prediction — Malaga

Final project for the SoSe 2026 Machine Learning course.


## Project structure

```
ML_Final_projet/
├── run_preprocess.py        # One command to build the modeling table
├── requirements.txt
├── data/
│   ├── raw/                 # Inside Airbnb downloads (not committed)
│   └── processed/           # Cleaned / merged datasets
├── notebooks/
│   └── airbnb_price_prediction.ipynb  # Full pipeline in one notebook
├── src/
│   ├── config.py            # Project settings & paths
│   ├── data.py              # Load raw files, clean, merge reviews
│   └── modeling.py          # Feature pipelines & metrics
├── models/                  # Saved trained models
└── reports/figures/         # Plots for presentations
```


