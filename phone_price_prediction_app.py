import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.title('Phone Price Prediction')

# Load the dataset
data = pd.read_csv('train.csv')
st.write('Data Preview:', data.head())

if st.checkbox('Preprocess Data'):
    target_column = st.selectbox('Select target column', data.columns)
    
    # Feature Engineering Steps
    data['sc'] = data['sc_h'] * data['sc_w']
    data['ram_n'] = data['ram'] * data['n_cores']
    data['px_res'] = data['px_height'] * data['px_width']
    data['px_res_n'] = data['px_height'] * data['px_width'] * data['n_cores']
    data.drop(['sc_h', 'sc_w', 'px_height', 'px_width', 'n_cores'], axis=1, inplace=True)
    
    data = data.dropna()
    data = pd.get_dummies(data)
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature engineering: Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model Training
    if st.checkbox('Train Models'):
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'CatBoost': CatBoostClassifier()
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        # Model Evaluation
        st.write('Model Evaluation:')
        for name, model in trained_models.items():
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f'{name} Accuracy: {accuracy:.2f}')

        # Prediction
        if st.checkbox('Make a Prediction'):
            st.write('Enter the features of the phone:')
            input_features = {}
            for feature in X.columns:
                input_features[feature] = st.text_input(f'Enter {feature}')
            
            if st.button('Predict'):
                input_df = pd.DataFrame([input_features])
                input_df = pd.get_dummies(input_df).reindex(columns=data.drop(target_column, axis=1).columns, fill_value=0)
                input_df = scaler.transform(input_df)
                
                predictions = {}
                for name, model in trained_models.items():
                    prediction = model.predict(input_df)
                    predictions[name] = prediction[0]
                
                st.write('Predictions:')
                for name, prediction in predictions.items():
                    st.write(f'{name}: {prediction}')

# Additional dataset for comparison or further analysis
test_data = pd.read_csv('test.csv')
st.write('Test Data Preview:', test_data.head(5))
