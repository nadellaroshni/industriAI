import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib

class HybridCreditScoring:
    def __init__(self):
        self.scaler = StandardScaler()
        self.gb_model = None
        self.nn_model = None
        self.feature_weights = {
            'credit_history': 0.30,
            'income_stability': 0.25,
            'repayment_capacity': 0.20,
            'banking_behavior': 0.15,
            'alternative_metrics': 0.10
        }
    
    def preprocess_data(self, data):
        """
        Preprocess the input data for both models
        """
        # Separate data sources according to weights
        traditional_features = [col for col in data.columns if col.startswith(('credit_', 'income_'))]
        alternative_features = [col for col in data.columns if col.startswith(('social_', 'utility_'))]
        realtime_features = [col for col in data.columns if col.startswith(('banking_', 'payment_'))]
        
        # Scale the features
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns
        )
        
        return scaled_data, traditional_features, alternative_features, realtime_features
    
    def build_gradient_boosting(self):
        """
        Build LightGBM model for feature importance and initial scoring
        """
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        self.gb_model = lgb.LGBMRegressor(**params)
    
    def build_neural_network(self, input_dim):
        """
        Build neural network for complex pattern analysis
        """
        inputs = layers.Input(shape=(input_dim,))
        
        # Deep neural network architecture
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.nn_model = Model(inputs=inputs, outputs=outputs)
        self.nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X, y, test_size=0.2):
        """
        Train both models
        """
        # Preprocess data
        X_processed, trad_feat, alt_feat, real_feat = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )
        
        # Train gradient boosting model
        self.build_gradient_boosting()
        self.gb_model.fit(X_train, y_train)
        
        # Get feature importance
        gb_predictions = self.gb_model.predict(X_processed)
        
        # Add GB predictions as a feature for NN
        X_processed_with_gb = pd.concat([
            X_processed,
            pd.Series(gb_predictions, name='gb_predictions')
        ], axis=1)
        
        # Train neural network
        self.build_neural_network(X_processed_with_gb.shape[1])
        self.nn_model.fit(
            X_processed_with_gb,
            y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return self.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Generate final credit score prediction
        """
        # Preprocess new data
        X_processed, _, _, _ = self.preprocess_data(X)
        
        # Get predictions from both models
        gb_pred = self.gb_model.predict(X_processed)
        X_processed_with_gb = pd.concat([
            X_processed,
            pd.Series(gb_pred, name='gb_predictions')
        ], axis=1)
        nn_pred = self.nn_model.predict(X_processed_with_gb)
        
        # Combine predictions with weighted average
        final_score = 0.4 * gb_pred + 0.6 * nn_pred.flatten()
        
        # Scale to credit score range (300-850)
        credit_score = 300 + (final_score * 550)
        return np.clip(credit_score, 300, 850)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def save_models(self, path):
        """
        Save both models and scaler
        """
        self.gb_model.save_model(f"{path}/gb_model.txt")
        self.nn_model.save(f"{path}/nn_model")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
    
    def load_models(self, path):
        """
        Load both models and scaler
        """
        self.gb_model = lgb.Booster(model_file=f"{path}/gb_model.txt")
        self.nn_model = tf.keras.models.load_model(f"{path}/nn_model")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
