import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

class Exercise1:
    """Housing price prediction using CNN models"""
    
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and preprocess housing data"""
        # Get the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data = pd.read_csv(os.path.join(script_dir, "Housing.csv"))
        
        # Convert categorical yes/no columns to numeric
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for col in [c for c in categorical_columns if c != "furnishingstatus"]:
            self.data[col] = self.data[col].map({'yes': 1, 'no': 0})

        # Convert furnishingstatus to numeric
        self.data['furnishingstatus'] = self.data['furnishingstatus'].map({
            'furnished': 1,
            'semi-furnished': 0.5,
            'unfurnished': 0
        })
        
        
        
        # Scale all numeric columns
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        
        """
        print("Data shape after preprocessing:", self.data.shape)
        print("Columns:", self.data.columns.tolist())
        print("Data types:\n", self.data.dtypes)
        """
        
    def split_univariate_sequence(self, sequence, n_steps):
        """Split data into samples for univariate prediction"""
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def split_multivariate_sequence(self, sequence, n_steps):
        """Split data for multivariate input prediction"""
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    

    def split_multiple_forecasting_sequence(self, sequence, n_steps):
        """Split data for multivariate parallel prediction"""
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def split_univariate_sequence_m_step(self,sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # encontrar el final de este patrón
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            
            # comprobar si estamos más allá de la secuencia
            if out_end_ix > len(sequence):
                break
            # reunir partes de entrada y salida del patrón
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    
    def split_multivariate_sequence_m_step(self, sequence: np.ndarray, n_steps_in: int, n_steps_out: int):
        X, y = list(), list()
        for i in range(len(sequence)):
            # encontrar el final de este patrón
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            
            # comprobar si estamos más allá de la secuencia
            if out_end_ix > len(sequence):
                break
            # reunir partes de entrada y salida del patrón
            seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def split_multivariate_sequence_m_step_parallel(self, sequence: np.ndarray, n_steps_in: int, n_steps_out: int):
        X, y = list(), list()
        for i in range(len(sequence)):
            # encontrar el final de este patrón
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            
            # comprobar si estamos más allá de la secuencia
            if out_end_ix > len(sequence):
                break
            # reunir partes de entrada y salida del patrón
            seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
        

    def create_cnn_model(self, input_shape, output_shape):
        """Create a basic CNN model"""
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, model_name):
        """Common training and evaluation code"""
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                          validation_split=0.1, verbose=0)
        
        # Evaluate model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f'{model_name} Test MSE: {test_loss}')
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        return test_loss, history

    def train_univariate_model(self):
        """1. Univariate CNN model"""
        prices = self.data['price'].values
        n_steps = 3
        X, y = self.split_univariate_sequence(prices, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.create_cnn_model(input_shape=(n_steps, 1), output_shape=1)
        
        return self.train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                                     "Univariate Model")

    def train_multivariate_input_model(self):
        """2.1 Multiple Input Series"""
        # Use all features except price as inputs
        features = self.data.drop('price', axis=1).values
        prices = self.data['price'].values
        
        n_steps = 3
        stack = np.column_stack((features, prices))
        X, y = self.split_multivariate_sequence(np.column_stack((features, prices)), n_steps)
        n_features = features.shape[1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.create_cnn_model(input_shape=(n_steps, n_features), output_shape=1)
        
        return self.train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                                     "Multivariate Input Model")

    def train_parallel_series_model(self):
        """2.2 Multiple Parallel Series"""
        data = self.data.values
        n_steps = 3
        
        X, y = self.split_multiple_forecasting_sequence(data, n_steps)
        n_features = data.shape[1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.create_cnn_model(input_shape=(n_steps, n_features), 
                                    output_shape=n_features)
        
        return self.train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                                     "Parallel Series Model")

    def train_multi_step_univariate_model(self):
        """3.1 Multi-step Univariate"""
        prices = self.data['price'].values
        n_steps_in, n_steps_out = 4, 2
        
        X, y = self.split_univariate_sequence_m_step(prices, n_steps_in, n_steps_out)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.create_cnn_model(input_shape=(n_steps_in, 1), 
                                    output_shape=n_steps_out)
        
        return self.train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                                     "Multi-step Univariate Model")

    def train_multi_step_multivariate_model(self):
        """3.2 Multi-step Multivariate"""
        features = self.data.drop('price', axis=1).values
        prices = self.data['price'].values
        n_steps_in, n_steps_out = 4, 2
        
        X, y = self.split_multivariate_sequence_m_step(
            np.column_stack((features, prices)), n_steps_in, n_steps_out)
        n_features = features.shape[1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.create_cnn_model(input_shape=(n_steps_in, n_features), 
                                    output_shape=n_steps_out)
        
        return self.train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                                     "Multi-step Multivariate Model")
    

    def train_multi_step_parallel_model(self):
        """3.3 Multi-step Multiple Parallel Series"""
        data = self.data.values
        n_steps_in, n_steps_out = 3, 2
        
        X, y = self.split_multivariate_sequence_m_step_parallel(data, n_steps_in, n_steps_out)
        n_features = data.shape[1]
        
        # Reshape y to match model output
        y = y.reshape(y.shape[0], -1)  # Flatten the output to match dense layer
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Update output shape to match flattened y
        output_shape = n_features * n_steps_out
        model = self.create_cnn_model(input_shape=(n_steps_in, n_features), 
                                    output_shape=output_shape)
        
        return self.train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                                     "Multi-step Parallel Model")

    def predict_and_plot_best_model(self, results):
        """Predict and plot results using the best model"""
        # Get the best model based on MSE
        mse_scores = {name: result[0] for name, result in results.items()}
        best_model_name = min(mse_scores, key=mse_scores.get)
        print(f"\nBest model: {best_model_name} with MSE: {mse_scores[best_model_name]}")
        
        # Get the data and model based on best performer
        if best_model_name == 'univariate':
            prices = self.data['price'].values
            n_steps = 3
            X, y = self.split_univariate_sequence(prices, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = self.create_cnn_model(input_shape=(n_steps, 1), output_shape=1)
        elif best_model_name == 'multivariate_input':
            features = self.data.drop('price', axis=1).values
            prices = self.data['price'].values
            n_steps = 3
            X, y = self.split_multivariate_sequence(np.column_stack((features, prices)), n_steps)
            n_features = features.shape[1]
            model = self.create_cnn_model(input_shape=(n_steps, n_features), output_shape=1)
        else:
            print("Prediction visualization not implemented for this model type")
            return
        
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual', marker='o')
        plt.plot(y_pred, label='Predicted', marker='*')
        plt.title(f'Actual vs Predicted House Prices using {best_model_name}')
        plt.xlabel('Sample')
        plt.ylabel('Price (scaled)')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)
        plt.pause(10)
        
        # Calculate and print metrics
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        r2 = 1 - (np.sum((y_test - y_pred.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        print("\nTest Set Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

    def main(self):
        """Main execution method"""
        print("Loading data")
        self.load_data()
        print("Data Loaded")
        
        # Dictionary to store results
        results = {}
        
        # Train all models
        print("Training univariate model")
        results['univariate'] = self.train_univariate_model()
        print("Training multivariate input model")
        results['multivariate_input'] = self.train_multivariate_input_model()
        print("Training parallel series model")
        results['parallel_series'] = self.train_parallel_series_model()
        print("Training multi-step univariate model")
        results['multi_step_univariate'] = self.train_multi_step_univariate_model()
        print("Training multi-step multivariate model")
        results['multi_step_multivariate'] = self.train_multi_step_multivariate_model()
        print("Training multi-step parallel model")
        results['multi_step_parallel'] = self.train_multi_step_parallel_model()
        
        # Compare results
        print("Comparing results")
        mse_scores = {name: result[0] for name, result in results.items()}
        results_df = pd.DataFrame(mse_scores, index=['MSE'])
        results_df = results_df.sort_values(by='MSE', axis=1, ascending=True)
        print("\nModel Performance Summary:")
        print(results_df)
        
        plt.figure(figsize=(12, 6))
        plt.bar(mse_scores.keys(), mse_scores.values())
        plt.title('MSE Comparison Across Models')
        plt.xticks(rotation=45)
        plt.ylabel('Mean Squared Error')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5)
        
        # Predict and plot using best model
        self.predict_and_plot_best_model(results)

class Exercise2:
    """Electricity demand prediction using CNN models"""
    
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and preprocess electricity data"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data = pd.read_csv(os.path.join(script_dir, "PRICE_AND_DEMAND_201801_NSW1.csv"))
        
    def split_univariate_sequence(self, sequence, n_steps):
        """Split data into samples for univariate prediction"""
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def train_univariate_model(self):
        """1. Univariate CNN model"""
        prices = self.data['TOTALDEMAND'].values
        
        # Scale the data
        prices = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        n_steps = 3
        X, y = self.split_univariate_sequence(prices, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Create and compile model
        model = keras.Sequential([
            keras.layers.Input(shape=(n_steps, 1)),
            keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                          validation_split=0.1, verbose=0)
        
        # Evaluate model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f'Univariate Model Test MSE: {test_loss}')
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Univariate Model - Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Univariate Model - Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Demand')
        plt.legend()
        plt.show(block=False)
        plt.pause(10)
        
        return test_loss, history

    def main(self):
        """Main execution method"""
        print("Loading data")
        self.load_data()
        print("Data Loaded")
        
        # Dictionary to store results
        results = {}
        
        # Train univariate model
        print("Training univariate model")
        test_loss, history = self.train_univariate_model()
        results['univariate'] = (test_loss, history)
        
        


       