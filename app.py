from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

app = Flask(__name__)

# Load and prepare dataset
KNN = pd.read_csv("diabetes.csv")
df = pd.DataFrame(KNN)

# Split features and target
x = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Hyperparameter tuning for optimal k with expanded range
param_grid = {'n_neighbors': np.arange(1, 30)}  # Increased range
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)  # 10-fold cross-validation
knn_cv.fit(x_train_scaled, y_train)

# Train the best model
best_k = knn_cv.best_params_['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train_scaled, y_train)

# Save model and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

@app.route('/')
def home():
    # Load model and scaler for metrics calculation
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Calculate metrics
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    # Pass all necessary data to the template
    return render_template('index.html', 
                           train_accuracy=train_accuracy if train_accuracy is not None else 0.0, 
                           test_accuracy=test_accuracy if test_accuracy is not None else 0.0,
                           train_report=train_report, 
                           test_report=test_report,
                           prediction=None)  # No prediction at the start

@app.route('/predict', methods=['POST'])
def predict():
    # Load model and scaler
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Get user input
    input_data = [
        int(request.form['pregnancies']),
        int(request.form['glucose']),
        int(request.form['blood_pressure']),
        int(request.form['skin_thickness']),
        int(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['pedigree']),
        int(request.form['age']),
    ]
    
    # Scale input data
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Calculate metrics again for display
    y_train_pred = model.predict(scaler.transform(x_train))
    y_test_pred = model.predict(scaler.transform(x_test))

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    return render_template('index.html', 
                           prediction=prediction[0], 
                           train_accuracy=train_accuracy if train_accuracy is not None else 0.0, 
                           test_accuracy=test_accuracy if test_accuracy is not None else 0.0,
                           train_report=train_report, 
                           test_report =test_report)

if __name__ == '__main__':
    app.run(debug=True)