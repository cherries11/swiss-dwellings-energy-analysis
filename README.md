# Swiss Dwellings Energy Analysis Project

## Project Overview

This project analyzes Swiss residential building data to predict energy consumption using architectural, geometric, and environmental features. The analysis demonstrates a complete data science pipeline from raw data to model evaluation, with emphasis on data preprocessing, feature engineering, and interpretability in the Swiss building context.

## Project Goals

- **Exploratory Data Analysis**: Understand dataset structure and data quality
- **Data Cleaning & Preprocessing**: Handle missing values and optimize data types  
- **Feature Engineering**: Create domain-relevant features based on building physics
- **Model Development**: Implement and compare tree-based models
- **Evaluation**: Assess preprocessing effectiveness through model performance

## Dataset Information

### Source
- **Dataset**: Swiss Dwellings from Zenodo
- **DOI**: 10.5281/zenodo.7070952
- **Provider**: Archilyse AG
- **Description**: Detailed data on 42,207 apartments (242,257 rooms) in 3,093 buildings including geometries and simulation results

### Files Used
1. **simulations.csv** (367 columns, 347,583 rows)
   - Room-level simulation data (layout, view, sun, noise, connectivity)
   - Aggregated metrics per area/room

2. **geometries.csv** (8 columns, 2,501,540 rows)  
   - Raw geometric elements (areas, walls, windows, doors, features)
   - Building-level geometric characteristics

### Key Features Categories
- **Layout**: Area, perimeter, compactness, window characteristics
- **View**: Visible buildings, greenery, sky, mountains (in steradians)
- **Sun**: Daylight simulation results at various times (kilolux)
- **Noise**: Traffic and train noise levels (dBA)
- **Connectivity**: Distances to key areas and centrality measures

## Project Structure

```
swiss_dwellings_analysis/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── run_pipeline.py                     # Main execution script
├── config.py                          # Configuration parameters
├── eda_pipeline.py                    # Exploratory Data Analysis
├── data_processing.py                 # Data loading and cleaning
├── feature_engineering.py             # Feature creation and selection
├── modeling.py                        # Model training and evaluation
├── visualization.py                   # Plot generation functions
├── utils.py                          # Helper functions and utilities
├── output/                           # Generated outputs (created automatically)
│   ├── models/                       # Saved model files
│   ├── data/                         # Processed datasets
│   ├── plots/                        # Visualization images
│   ├── reports/                      # Analysis reports in JSON
│   └── model_comparison/             # Model evaluation results
└── documentation/                    # Project documentation
    └── complete_documentation.md     # Detailed project documentation
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB disk space for data and outputs

### Installation Steps

1. Clone or download the project files
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download dataset files from Zenodo and place in project root:
   - simulations.csv
   - geometries.csv

### Required Packages
- pandas>=1.5.0
- numpy>=1.23.0
- scikit-learn>=1.2.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- joblib>=1.2.0
- scipy>=1.9.0

## Usage Instructions

### Quick Start

Run the complete analysis pipeline:

```bash
python run_pipeline.py
```

This executes all steps:
1. Data loading and memory optimization
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Feature engineering
5. Model training and comparison
6. Results generation and saving

### Running Individual Components

#### Exploratory Data Analysis Only
```bash
python eda_pipeline.py
```

#### Data Processing Only
```bash
python data_processing.py
```

#### Model Training Only
```bash
python modeling.py
```

### Configuration Options

Modify `config.py` to adjust analysis parameters:

```python
CONFIG = {
    'data_path': './',                    # Path to CSV files
    'sample_size': 100000,                # Number of rows to sample (None for all)
    'test_size': 0.2,                     # Test set proportion
    'random_state': 42,                   # Random seed for reproducibility
    'variance_threshold': 0.01,           # Feature variance filter
    'max_features': 15,                   # Maximum features for modeling
    'output_dir': './output'              # Output directory
}
```

## Methodology Summary

### Data Processing Pipeline

1. **Memory-Efficient Loading**: Optimized dtype assignment and chunked reading
2. **Apartment Aggregation**: Room-level data aggregated to 11,913 apartments
3. **Missing Value Handling**: Tiered strategy based on missingness percentage
4. **Feature Engineering**: Created domain-specific features (area-window ratio, solar efficiency, etc.)
5. **Target Creation**: Synthetic energy consumption based on Swiss building physics

### Modeling Approach

- **Models Compared**: RandomForest, GradientBoosting, Ridge (baseline)
- **Preprocessing**: Median imputation, standard scaling, one-hot encoding
- **Evaluation**: 5-fold cross-validation with R², MAE, MAPE metrics
- **Feature Selection**: Variance threshold and correlation-based selection

## Key Results

### Model Performance
| Model | Test R² | Test MAE | Test MAPE | Training Time |
|-------|---------|----------|-----------|---------------|
| RandomForest | 0.6180 | 150 kWh/year | 3.9% | ~45s |
| GradientBoosting | 0.6173 | 152 kWh/year | 4.0% | ~60s |
| Ridge | 0.4351 | 291 kWh/year | 8.5% | ~5s |

### Key Insights
1. **Top Predictors**: Layout perimeter (56.7%), layout area (17.6%), window perimeter (5.1%)
2. **Swiss Context**: Results align with building physics principles (envelope heat loss critical)
3. **Data Quality**: Successfully handled 4.17% missing data across 367 features
4. **Memory Efficiency**: Reduced memory usage by 1.1% through dtype optimization

## Output Files

### Generated Outputs (in `output/` directory)

#### Models
- `best_model_randomforest_YYYYMMDD_HHMMSS.pkl` - Best performing model with metadata

#### Processed Data
- `swiss_dwellings_processed_final_YYYYMMDD_HHMMSS.csv` - Cleaned dataset with target

#### Visualizations
- `energy_distribution_improved.png` - Target variable distribution
- `feature_correlations_enhanced.png` - Top correlated features
- `model_comparison.png` - Model performance comparison
- `feature_importance_*.png` - Feature importance plots

#### Reports
- `final_report_comprehensive_*.json` - Complete analysis summary
- `data_quality_report.json` - Data cleaning documentation
- `model_comparison_results_*.json` - Model evaluation metrics
- `feature_importance_*.csv` - Detailed importance scores

## Technical Details

### Memory Optimization Techniques
- Dtype downcasting (float64 → float32, int64 → int32)
- Categorical conversion for low-cardinality features
- Chunked file reading for large datasets
- Explicit garbage collection between pipeline stages

### Data Cleaning Strategy
1. **High missing (>90%)**: Feature removal (7 features)
2. **Moderate missing (10-90%)**: Median/mode imputation
3. **Low missing (<10%)**: Same as moderate, preserved for analysis
4. **Outliers**: Winsorization at 1.5×IQR bounds

### Feature Engineering
- **Area-to-window ratio**: Heat loss/gain estimation
- **Solar efficiency**: Passive heating potential
- **Compactness squared**: Non-linear efficiency relationship
- **Altitude factor**: Climate adjustment for Swiss topography

## Limitations and Assumptions

### Data Limitations
- Synthetic target variable (no actual energy consumption data)
- Commercial client bias in sample
- Static snapshot without temporal variation

### Modeling Assumptions
- Missing data is missing at random
- Apartment-level aggregation appropriate for energy analysis
- Feature relationships stable across building types

### Technical Constraints
- Memory limitations require sampling for full exploration
- Simplified building physics in target generation
- No uncertainty quantification in predictions

## Reproducibility

### Random Seeds
All random processes use `random_state=42` for reproducibility:
- Data sampling
- Train-test splitting
- Model initialization
- Cross-validation folds

### Version Control
- Python package versions specified in requirements.txt
- Timestamped output files prevent overwriting
- Complete configuration in config.py

### Execution Environment
Tested on:
- Python 3.8-3.11
- Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- 4-16GB RAM systems

## Future Improvements

### Data Enhancements
1. Incorporate actual energy consumption data
2. Add building material and insulation information
3. Include seasonal and temporal variation
4. Expand geographic coverage across Switzerland

### Model Improvements
1. Hyperparameter tuning with grid/random search
2. Ensemble methods combining multiple models
3. Uncertainty quantification with Bayesian methods
4. Deep learning approaches for complex patterns

### Application Extensions
1. Retrofit recommendation system
2. Building design optimization tool
3. Policy impact simulation
4. Educational platform for building physics

## Citation and References

### Dataset Citation
```
Swiss Dwellings: A large dataset of apartment models including aggregated 
geolocation-based simulation results covering viewshed, natural light, 
traffic noise, centrality and geometric analysis. Zenodo. 
https://doi.org/10.5281/zenodo.7070952
```

### Technical References
- SIA 380/1: Swiss standard for building energy calculation
- Scikit-learn documentation for model implementations
- Pandas documentation for data processing techniques

## Support and Contact

For questions or issues:
1. Check the complete documentation in `documentation/complete_documentation.md`
2. Review generated error reports in `output/reports/`
3. Ensure all dependencies are installed correctly
4. Verify dataset files are in the correct location

## License

This project is for academic/educational purposes. Dataset usage should comply with Zenodo's terms and conditions. Code is provided as-is without warranty.

## Acknowledgments

- Archilyse AG for providing the dataset
- Swiss building physics community for domain context
- Open-source Python community for analysis tools

---
