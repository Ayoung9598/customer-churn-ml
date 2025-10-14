# Customer Churn Prediction

Production-ready Python project for training and serving a customer churn classifier using scikit-learn and XGBoost. Includes modular data processing, feature engineering, model training/evaluation, and both FastAPI REST service and Streamlit UI for predictions.

## Project Structure

```
customer-churn-ml/
├─ src/                          # Source code modules
│  ├─ data/                      # Data loading and preprocessing
│  │  └─ processing.py           # CSV loading, target normalization, train/val split
│  ├─ features/                  # Feature engineering pipeline
│  │  └─ engineering.py          # Preprocessing transformers and feature selection
│  ├─ models/                    # Model training and evaluation
│  │  └─ train.py                # Classifier builders, training, and model persistence
│  ├─ evaluation/                # Model evaluation utilities
│  │  └─ metrics.py              # Classification metrics and calibration
│  └─ utils/                     # Common utilities
│     └─ io.py                   # JSON/model save/load helpers
├─ notebooks/                    # Exploratory data analysis
│  └─ EDA.ipynb                  # Jupyter notebook template for data exploration
├─ data/                         # Raw datasets (place your CSV here)
├─ artifacts/                    # Trained models and metrics (auto-generated)
├─ configs/                      # Configuration files
│  └─ columns.yaml               # Column definitions for Telco dataset
├─ main.py                       # CLI training script
├─ app.py                        # FastAPI REST service
├─ streamlit_app.py              # Streamlit web UI
├─ requirements.txt              # Python dependencies
├─ .gitignore                    # Git ignore rules
└─ README.md                     # This file
```

## Detailed File Documentation

### Core Training & API Files

#### `main.py` - Command Line Training Interface
**Purpose**: Orchestrates the complete ML training pipeline from data loading to model saving.

**Key Functions**:
- `parse_args()`: Parses command-line arguments for data path, target column, feature columns, model type, etc.
- `main()`: Loads data, builds feature pipeline, trains model, and saves results.

**Usage**:
```bash
python main.py --data data/churn_data.csv --config configs/columns.yaml --model xgb
```

**Features**:
- Supports both explicit column lists via CLI flags and YAML config files
- CLI arguments override config file values
- Handles target column normalization (Yes/No → 1/0)
- Saves trained pipeline and metrics to `artifacts/`

#### `app.py` - FastAPI REST Service
**Purpose**: Exposes the trained model as a REST API for real-time predictions.

**Key Components**:
- `PredictRequest`: Pydantic model for input validation
- `PredictResponse`: Pydantic model for output formatting
- `/healthz`: Health check endpoint
- `/predict`: Main prediction endpoint

**Features**:
- Loads model on startup from `artifacts/churn_model.joblib`
- Handles batch predictions (multiple records)
- Returns both probabilities and binary predictions
- Built-in API documentation at `/docs`

#### `streamlit_app.py` - Interactive Web UI
**Purpose**: Provides a user-friendly web interface for model predictions.

**Key Features**:
- **Single Prediction Tab**: Form inputs for individual customer predictions
- **Batch Prediction Tab**: CSV upload for bulk predictions with download
- **Configuration Sidebar**: Model path, config file, and sample CSV settings
- **Auto-populated Choices**: Infers categorical options from sample data

**UI Components**:
- Numeric inputs with sensible defaults
- Dropdown selectors for categorical features
- Real-time prediction results
- CSV download for batch results

### Source Code Modules (`src/`)

#### `src/data/processing.py` - Data Pipeline
**Purpose**: Handles data loading, preprocessing, and train/validation splitting.

**Key Functions**:
- `_normalize_binary_target()`: Converts string binary labels (Yes/No, True/False) to 1/0
- `load_csv_dataset()`: Loads CSV, extracts features and target, normalizes target
- `coerce_numeric_frame()`: Converts string numbers to numeric, handles missing values
- `create_preprocessor()`: Builds scikit-learn preprocessing pipeline
- `train_valid_split()`: Creates stratified train/validation splits

**Preprocessing Pipeline**:
- **Numeric Features**: Coercion → Median imputation → Standard scaling
- **Categorical Features**: Most frequent imputation → One-hot encoding
- **Missing Value Handling**: Automatic imputation for both numeric and categorical

#### `src/features/engineering.py` - Feature Engineering
**Purpose**: Combines preprocessing with optional feature selection.

**Key Functions**:
- `build_feature_pipeline()`: Creates end-to-end feature pipeline
- Integrates preprocessing with `SelectKBest` feature selection
- Uses mutual information for feature scoring

**Pipeline Steps**:
1. Preprocessing (imputation, scaling, encoding)
2. Optional feature selection (SelectKBest with mutual_info_classif)
3. Output: Transformed feature matrix ready for modeling

#### `src/models/train.py` - Model Training
**Purpose**: Model building, training, evaluation, and persistence.

**Key Functions**:
- `build_classifier()`: Factory function for creating classifiers
  - **Logistic Regression**: `logreg`, `logistic`, `logistic_regression`
  - **XGBoost**: `xgb`, `xgboost` with optimized hyperparameters
- `evaluate_binary_classifier()`: Computes ROC AUC and AUC metrics
- `fit_and_save()`: Trains pipeline, evaluates, saves model and metrics

**Supported Models**:
- **Logistic Regression**: Fast, interpretable baseline
- **XGBoost**: Gradient boosting with hyperparameter tuning

**Output Files**:
- `{name}.joblib`: Complete trained pipeline
- `{name}_metrics.json`: Performance metrics

#### `src/evaluation/metrics.py` - Model Evaluation
**Purpose**: Comprehensive model evaluation utilities.

**Key Functions**:
- `compute_classification_report()`: Precision, recall, F1-score
- `compute_confusion_matrix()`: Confusion matrix for binary classification
- `compute_calibration()`: Calibration curve for probability assessment

#### `src/utils/io.py` - I/O Utilities
**Purpose**: Common file I/O operations for JSON and model persistence.

**Key Functions**:
- `save_json()` / `load_json()`: JSON file operations
- `save_model()` / `load_model()`: Model persistence using joblib

### Configuration & Data Files

#### `configs/columns.yaml` - Column Configuration
**Purpose**: Defines feature columns for the Telco Customer Churn dataset.

**Structure**:
```yaml
target: Churn                    # Binary target column
numeric:                         # Numeric features
  - tenure
  - MonthlyCharges
  - TotalCharges
categorical:                     # Categorical features
  - gender
  - SeniorCitizen
  - Partner
  # ... (complete list)
```

**Benefits**:
- Avoids long CLI argument lists
- Centralized column management
- Easy to modify for different datasets

#### `notebooks/EDA.ipynb` - Exploratory Data Analysis
**Purpose**: Jupyter notebook template for data exploration.

**Sections**:
- Data loading and basic info
- Missing value analysis
- Target distribution
- Feature correlation analysis
- Visualization templates

#### `requirements.txt` - Dependencies
**Purpose**: Pinned Python package versions for reproducible environments.

**Categories**:
- **Core ML**: numpy, pandas, scikit-learn, xgboost, joblib
- **API**: fastapi, uvicorn, pydantic
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: jupyterlab, streamlit
- **Utilities**: pyyaml, python-dotenv

#### `.gitignore` - Version Control
**Purpose**: Excludes generated files and sensitive data from Git.

**Excluded**:
- Python artifacts (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`)
- Model artifacts (`artifacts/`)
- IDE files (`.vscode/`, `.idea/`)
- Environment files (`.env`, `.kaggle/`)

## Setup Instructions

### 1. Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (Git Bash)
source venv/Scripts/activate
```

### 2. Install Dependencies
```powershell
# Upgrade pip
pip install --upgrade pip

# Install with timeout for slow networks
pip install --default-timeout=180 -r requirements.txt
```

### 3. Prepare Data
- Place your CSV dataset in `data/` folder
- Ensure it has a binary target column (e.g., `Churn` with Yes/No values)
- Update `configs/columns.yaml` to match your dataset schema

## Usage Examples

### Training with Config File
```bash
python main.py --data data/churn_data.csv --config configs/columns.yaml --model xgb --out artifacts --name churn_model
```

### Training with Explicit Columns
```bash
python main.py --data data/churn_data.csv --target Churn \
  --num-cols tenure MonthlyCharges TotalCharges \
  --cat-cols gender SeniorCitizen Partner Dependents \
  --model xgb --k-best 50
```

### FastAPI Service
```bash
# Start server
uvicorn app:app --reload --port 8000

# Health check
curl http://127.0.0.1:8000/healthz

# Predict
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"tenure":12,"MonthlyCharges":75.5,"TotalCharges":900.0,"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Credit card"}]}'
```

### Streamlit UI
```bash
# Launch web interface
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

## API Endpoints

### FastAPI Endpoints

#### `GET /healthz`
**Purpose**: Health check endpoint
**Response**: `{"status": "ok"}` or `{"status": "model_not_loaded"}`

#### `POST /predict`
**Purpose**: Make predictions on customer data
**Request Body**:
```json
{
  "records": [
    {
      "tenure": 12,
      "MonthlyCharges": 75.5,
      "TotalCharges": 900.0,
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Credit card"
    }
  ]
}
```

**Response**:
```json
{
  "probabilities": [0.27],
  "predictions": [0]
}
```

## Model Performance

The trained XGBoost model typically achieves:
- **ROC AUC**: ~0.83 (on Telco Customer Churn dataset)
- **Features**: 19 features (3 numeric, 16 categorical)
- **Preprocessing**: Automatic imputation, scaling, one-hot encoding

## Development Guidelines

### Adding New Models
1. Extend `build_classifier()` in `src/models/train.py`
2. Add model name to CLI choices in `main.py`
3. Update hyperparameters as needed

### Custom Preprocessing
1. Modify `create_preprocessor()` in `src/data/processing.py`
2. Add new transformers to the pipeline
3. Update column handling if needed

### Feature Engineering
1. Extend `build_feature_pipeline()` in `src/features/engineering.py`
2. Add feature selection methods
3. Implement custom transformers

### Evaluation Metrics
1. Add new metrics to `src/evaluation/metrics.py`
2. Update `evaluate_binary_classifier()` in `src/models/train.py`
3. Modify output format if needed

## Troubleshooting

### Common Issues

#### "Cannot use median strategy with non-numeric data"
- **Cause**: Numeric columns contain string values (e.g., blank spaces)
- **Solution**: The preprocessing pipeline now includes automatic numeric coercion

#### "Model not loaded" in API
- **Cause**: Model file doesn't exist at expected path
- **Solution**: Train model first or update model path in `app.py`

#### Pickling errors during training
- **Cause**: Local functions in preprocessing pipeline
- **Solution**: All functions are now top-level for proper serialization

#### Import errors
- **Cause**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt`

### Performance Tips

1. **Use config files** instead of long CLI arguments
2. **Start with Logistic Regression** for faster initial testing
3. **Use feature selection** (`--k-best`) for high-dimensional data
4. **Monitor memory usage** with large datasets

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue with detailed error information

**Contact**: abiolateslim1@gmail.com