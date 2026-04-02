import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

def train():
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv('car_data/car.data', names=columns)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define exact categories for accurate ordinal encoding and compatibility
    categories = [
        ['vhigh', 'high', 'med', 'low'], # buying
        ['vhigh', 'high', 'med', 'low'], # maint
        ['2', '3', '4', '5more'],        # doors
        ['2', '4', 'more'],              # persons
        ['small', 'med', 'big'],         # lug_boot
        ['low', 'med', 'high']           # safety
    ]
    
    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    pipeline = Pipeline([
        ('encoder', encoder),
        ('model', rf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("Classification Report on Test Data:")
    print(classification_report(y_test, y_pred))
    
    with open('car_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Pipeline model saved successfully to car_model.pkl")

if __name__ == '__main__':
    train()
