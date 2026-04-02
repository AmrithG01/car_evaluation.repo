# Car Evaluation Prediction Interface 🚗

This project provides an interactive web interface to predict the acceptability rating of a car based on its attributes. This is built using a machine learning model trained on the classic Car Evaluation Dataset, using scikit-learn and Streamlit.

## Features
- **Predictive AI**: A Random Forest Classifier that provides accurate predictions on car acceptability (Unacceptable, Acceptable, Good, Very Good).
- **Interactive UI**: A beautiful, user-friendly webpage built with Streamlit.
- **Fast Predictions**: Instantaneous results upon selecting car attributes.

## Project Structure
- `app.py`: The main Streamlit web application.
- `train_model.py`: Script to train the machine learning pipeline and save the model as `car_model.pkl`.
- `main.ipynb`: Jupyter Notebook with data exploration and model evaluations.
- `car_data/`: Directory containing the Car Evaluation dataset (`car.data`).
- `car_model.pkl`: The serialized pre-trained model pipeline.

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model (Optional)
If you want to re-train the model on your own or if `car_model.pkl` is missing:
```bash
python train_model.py
```
This will read the data from `car_data/car.data`, train the Random Forest pipeline, print out the classification report, and generate a new `car_model.pkl`.

### 2. Running the Web App
Start the Streamlit application with the following command:
```bash
streamlit run app.py
```
This will spin up a local server and automatically open the application in your default web browser.

## Technologies Used
- Python
- pandas
- scikit-learn
- Streamlit
- Jupyter

## Dataset Attributes
The model predicts car acceptability based on the following features:
- **Buying Price** (vhigh, high, med, low)
- **Maintenance Price** (vhigh, high, med, low)
- **Number of Doors** (2, 3, 4, 5more)
- **Capacity (Persons)** (2, 4, more)
- **Luggage Boot Size** (small, med, big)
- **Estimated Safety** (low, med, high)
