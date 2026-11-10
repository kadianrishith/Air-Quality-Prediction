Air Quality Prediction Using Machine Learning
Project Overview

Air Quality Prediction is a Python-based machine learning project that estimates Carbon Monoxide (CO) concentration using historical air pollution and environmental measurements.

The project uses a Random Forest Regression algorithm to learn patterns from environmental data and predict CO concentration.

An interactive web application is developed using Streamlit, allowing users to enter environmental measurements and obtain predictions.

Objectives
Analyze historical air quality data.
Clean and preprocess environmental measurements.
Explore relationships between different air pollutants.
Train a machine learning regression model.
Evaluate model performance using appropriate metrics.
Develop an interactive prediction application.
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit
Joblib
Dataset

The project uses the UCI Air Quality Dataset.

Dataset source: https://archive.ics.uci.edu/dataset/360/air+quality

The dataset contains hourly air pollution measurements and environmental information.

Machine Learning Algorithm

Random Forest Regression

The model uses the following input features:

PT08.S1(CO)
NMHC(GT)
C6H6(GT)
NOx(GT)
NO2(GT)
Temperature
Relative Humidity
Absolute Humidity

Prediction Target: CO(GT) – Carbon Monoxide concentration.

Model Evaluation

Initial model evaluation results:

Metric	Result
Mean Absolute Error (MAE)	0.388
Root Mean Squared Error (RMSE)	0.568
R² Score	0.822

These are the initial results obtained on the chronological test split. Performance should be independently verified when reproducing the training process.

Project Structure
text
Air-Quality-Prediction/
│
├── dataset/
├── notebooks/
├── models/
├── outputs/
│
├── train_model.py
├── preprocess.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

Installation and Setup
1. Clone the repository
bash
git clone https://github.com/kadianrishith/Air-Quality-Prediction.git

2. Navigate to the project folder
bash
cd Air-Quality-Prediction

3. Create a virtual environment
bash
python -m venv venv

4. Activate the environment on Windows
powershell
.\venv\Scripts\Activate.ps1

5. Install dependencies
bash
python -m pip install -r requirements.txt

6. Run data analysis
bash
python notebooks/data_analysis.py

7. Run data preprocessing
bash
python preprocess.py

8. Train the machine learning model
bash
python train_model.py

9. Launch the Streamlit application
bash
python -m streamlit run app.py


Open the local URL displayed in your terminal to access the application.

Application Features
Environmental measurement inputs.
Machine learning-based CO concentration estimation.
Interactive Streamlit interface.
Model evaluation visualizations.
Data analysis and preprocessing.
Limitations

This application estimates CO concentration from simultaneous environmental measurements. It does not provide future pollution forecasting or an official Air Quality Index (AQI) classification.

Predictions depend on the quality and range of the input measurements and the training dataset.

Future Enhancements
Predict future air pollution levels.
Add official AQI calculation using appropriate pollutant measurements and standards.
Integrate real-time air quality APIs.
Add more machine learning algorithms for comparison.
Improve the dashboard with additional visualizations.
Author

Developed as part of a Python Machine Learning Internship Project.

Dataset Attribution

Dataset: UCI Air Quality Dataset, UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/360/air+quality
