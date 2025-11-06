# Data Directory

This directory is used to cache downloaded datasets used in Aurora-GLM examples.

## How It Works

- **Basic examples** (quickstart, regression, classification, etc.): Generate synthetic data directly in notebooks - no downloads needed
- **Case study examples**: Automatically download real datasets on first run and cache them here
- **Cached files**: Downloaded once, reused on subsequent runs
- **Git ignored**: All data files (*.csv, *.zip, etc.) are ignored by git

## Directory Structure (After Running Examples)

```
data/
├── .gitignore                 # Ignores all data files
├── README.md                  # This file
├── insurance.csv              # Medical insurance charges (CC0)
├── sleepstudy.csv            # Sleep deprivation study (GPL-2)
├── airquality.csv            # Air quality measurements (GPL-3)
└── [other cached files]
```

## Data Sources

All datasets use permissive open licenses:

| Dataset | Source | License | Size | Use Case |
|---------|--------|---------|------|----------|
| `insurance.csv` | [Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets) | CC0 | ~60 KB | Gamma GLM |
| `sleepstudy.csv` | [lme4 R package](https://github.com/lme4/lme4) | GPL-2 | ~5 KB | Longitudinal GAMM |
| `airquality.csv` | [R datasets](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html) | GPL-3 | ~4 KB | GAM smoothing |

## Clearing Cache

To force re-download of all datasets:

```bash
# From examples/ directory
rm data/*.csv
```

Then re-run notebooks - they will automatically re-download.

## Manual Download

If automatic download fails (e.g., no internet), you can manually download:

### Insurance Dataset
```bash
curl -o data/insurance.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
```

### Sleep Study Dataset
```bash
curl -o data/sleepstudy.csv https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/sleepstudy.csv
```

### Air Quality Dataset
```bash
curl -o data/airquality.csv https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv
```

## Citations

If you use these datasets in publications, please cite:

**Insurance Dataset:**
- Source: Machine Learning with R datasets collection
- License: CC0 (Public Domain)

**Sleep Study Dataset:**
- Belenky et al. (2003). "Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study." Journal of Sleep Research, 12(1), 1-12.
- Distributed via lme4 R package: Bates, Mächler, Bolker, Walker (2015)

**Air Quality Dataset:**
- Chambers, Cleveland, Kleiner, and Tukey (1983). Graphical Methods for Data Analysis. Wadsworth.
- Distributed via R datasets package: R Core Team

## License

- Cached data files: Follow their respective source licenses (see table above)
- Data loading code (`utils/data_loader.py`): MIT License (same as Aurora-GLM)

## Troubleshooting

**Download fails:**
- Check internet connection
- Try manual download (see above)
- Check if source URLs are still active

**Permission errors:**
- Ensure write permissions in `examples/data/` directory
- On Unix: `chmod 755 examples/data`

**Disk space:**
- All datasets combined: < 1 MB
- If disk space is limited, delete unused `.csv` files
