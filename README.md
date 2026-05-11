# Diabetes Prediction Web App

A machine learning web application that predicts diabetes risk using the K-Nearest Neighbors algorithm. Built with Flask and scikit-learn, this application provides an interactive interface for healthcare professionals and individuals to assess diabetes likelihood based on various health metrics.

## Features

- **Interactive Web Interface**: User-friendly form for inputting patient health data
- **Real-time Predictions**: Instant diabetes risk assessment
- **Model Performance Metrics**: Displays accuracy, precision, recall, and F1-scores
- **Medical Guidance**: Includes normal value ranges for each health metric
- **Optimized ML Model**: Uses hyperparameter tuning with cross-validation for best performance

## Dataset

This project uses the Pima Indians Diabetes Dataset, which contains the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (mg/dL)
- Blood Pressure: Diastolic blood pressure (mm Hg)
- Skin Thickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- Diabetes Pedigree Function: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 = no diabetes, 1 = diabetes)

## Model Details

- **Algorithm**: K-Nearest Neighbors Classifier
- **Hyperparameter Tuning**: Grid search with 10-fold cross-validation (k values 1-29)
- **Feature Scaling**: StandardScaler for normalization
- **Train/Test Split**: 70/30 ratio with random_state=42 for reproducibility

## Installation

1. **Clone or download the project**:
   ```bash
   cd "your-project-directory"
   ```

2. **Install Python dependencies**:
   ```bash
   pip install flask pandas numpy scikit-learn joblib
   ```

3. **Ensure the dataset file is present**:
   - The `diabetes.csv` file should be in the root directory

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Input patient data** in the form fields and click "Predict"

4. **View results**:
   - Prediction outcome (0 = No Diabetes, 1 = Diabetes)
   - Model performance metrics
   - Detailed classification reports

## Project Structure

```
diabetes-prediction/
│
├── app.py                    # Main Flask application
├── diabetes.csv             # Dataset file
├── knn_model.pkl           # Trained model (generated)
├── scaler.pkl              # Feature scaler (generated)
│
├── static/
│   └── style.css           # CSS styling
│
├── templates/
│   └── index.html          # Main web template
│
└── README.md               # This file
```

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS
- **Model Persistence**: joblib

## Model Performance

The KNN model is trained and evaluated on the diabetes dataset. Performance metrics are displayed on the web interface including:
- Training and test accuracy
- Precision, recall, and F1-scores for both classes
- Classification reports

## Contributing

This is an educational project. Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Optimize the model

## License

This project is for educational purposes. Please ensure compliance with data usage policies and medical ethics when using this application.

## Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.