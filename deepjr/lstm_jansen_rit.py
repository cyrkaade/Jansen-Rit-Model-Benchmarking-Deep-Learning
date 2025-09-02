import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os


class JRLSTMModel:
    def __init__(self, 
                 num_channels=1, 
                 num_timepoints=1000, 
                 estim_params=('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C')):
        """
        LSTM-based model for Jansen-Rit parameter estimation
        
        Args:
            num_channels: Number of EEG channels/features
            num_timepoints: Number of time points in each ERP
            estim_params: Tuple of parameter names to estimate
        """
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        self.estim_params = estim_params
        self.output_dim = len(estim_params)
        self.model = None
        self.history = None
        self.param_stats = None  # To store normalization statistics
        
    def build_model(self, 
                    lstm_units=128, 
                    dense_units=64, 
                    dropout_rate=0.3, 
                    learning_rate=0.001):
        """
        Build the LSTM model architecture
        
        This model combines CNNs for feature extraction with bidirectional LSTMs
        for capturing temporal dependencies in the ERP signals.
        """
        # Input layer
        inputs = Input(shape=(self.num_channels, self.num_timepoints))
        
        # Reshape for CNN layers
        reshaped = tf.keras.layers.Reshape((self.num_timepoints, self.num_channels))(inputs)
        
        # CNN feature extraction
        conv1 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(reshaped)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(pool2)
        pool3 = MaxPooling1D(pool_size=2)(conv3)
        
        # LSTM layers
        lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(pool3)
        drop1 = Dropout(dropout_rate)(lstm1)
        lstm2 = LSTM(lstm_units)(drop1)
        drop2 = Dropout(dropout_rate)(lstm2)
        
        # Dense layers
        dense1 = Dense(dense_units*2, activation='relu')(drop2)
        drop3 = Dropout(dropout_rate)(dense1)
        dense2 = Dense(dense_units, activation='relu')(drop3)
        drop4 = Dropout(dropout_rate/2)(dense2)
        
        # Output layer
        outputs = Dense(self.output_dim, activation='linear')(drop4)
        
        # Build the model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def normalize_params(self, y_data):
        """
        Normalize the parameter values for better training
        
        Args:
            y_data: Parameter values array or dataframe
            
        Returns:
            Normalized parameters
        """
        if self.param_stats is None:
            # First time, calculate stats
            self.param_stats = {
                'mean': np.mean(y_data, axis=0),
                'std': np.std(y_data, axis=0)
            }
        
        # Normalize (z-score)
        y_norm = (y_data - self.param_stats['mean']) / self.param_stats['std']
        
        return y_norm
    
    def denormalize_params(self, y_norm):
        """
        Convert normalized parameters back to original scale
        
        Args:
            y_norm: Normalized parameter values
            
        Returns:
            Original scale parameters
        """
        if self.param_stats is None:
            raise ValueError("Cannot denormalize parameters before normalization")
        
        y_original = y_norm * self.param_stats['std'] + self.param_stats['mean']
        
        return y_original
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=100, batch_size=32, patience=20, 
                   model_path='best_lstm_model.h5'):
        """
        Train the LSTM model
        
        Args:
            X_train: Training ERP data [samples, channels, time points]
            y_train: Training parameter values [samples, parameters]
            X_val: Validation ERP data
            y_val: Validation parameter values
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            model_path: Path to save best model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Normalize parameters
        y_train_norm = self.normalize_params(y_train)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_norm = self.normalize_params(y_val)
            validation_data = (X_val, y_val_norm)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if validation_data else 'loss', 
                          patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', 
                              factor=0.5, patience=patience//2, min_lr=1e-6)
        ]
        
        # Add model checkpoint if path is provided
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss' if validation_data else 'loss',
                               save_best_only=True, save_weights_only=False)
            )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train_norm,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X_data, denormalize=True):
        """
        Make predictions with the model
        
        Args:
            X_data: Input ERP data
            denormalize: Whether to convert predictions to original scale
            
        Returns:
            Predicted parameter values
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Make predictions
        y_pred_norm = self.model.predict(X_data)
        
        # Denormalize if requested
        if denormalize and self.param_stats is not None:
            return self.denormalize_params(y_pred_norm)
        
        return y_pred_norm
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test: Test ERP data
            y_test: True parameter values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        
        # Individual parameter MSEs
        param_mses = {}
        for i, param in enumerate(self.estim_params):
            param_mses[param] = mean_squared_error(y_test[:, i], y_pred[:, i])
        
        return {
            'mse': mse,
            'param_mses': param_mses
        }
    
    def print_correlations(self, X_test, y_test):
        """
        Print correlation coefficients between predictions and true values
        
        Args:
            X_test: Test ERP data
            y_test: True parameter values
            
        Returns:
            DataFrame with correlation coefficients
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate correlations for each parameter
        correlations = {}
        r2_scores = {}
        
        for i, param in enumerate(self.estim_params):
            # Correlation coefficient
            corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
            correlations[param] = corr
            
            # R^2 score
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            r2_scores[param] = r2
            
            print(f"{param}: Correlation = {corr:.4f}, R² = {r2:.4f}")
        
        # Create a DataFrame for easier visualization
        results = pd.DataFrame({
            'Parameter': list(self.estim_params),
            'Correlation': [correlations[p] for p in self.estim_params],
            'R²': [r2_scores[p] for p in self.estim_params]
        })
        
        return results
    
    def plot_test_regressions(self, X_test, y_test, num_samples=None):
        """
        Plot predictions vs actual values for each parameter
        
        Args:
            X_test: Test ERP data
            y_test: True parameter values
            num_samples: Number of samples to plot (None = all)
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Determine samples to plot
        if num_samples is None or num_samples > len(y_test):
            num_samples = len(y_test)
        
        sample_indices = np.random.choice(len(y_test), num_samples, replace=False)
        
        # Plot each parameter
        n_params = len(self.estim_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        for i, param in enumerate(self.estim_params):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Get actual and predicted values for this parameter
            actual = y_test[sample_indices, i]
            predicted = y_pred[sample_indices, i]
            
            # Plot diagonal line (perfect predictions)
            min_val = min(np.min(actual), np.min(predicted))
            max_val = max(np.max(actual), np.max(predicted))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Plot predictions
            plt.scatter(actual, predicted, alpha=0.5)
            
            # Calculate and show correlation
            corr = np.corrcoef(actual, predicted)[0, 1]
            r2 = r2_score(actual, predicted)
            
            plt.title(f"{param}: Corr = {corr:.4f}, R² = {r2:.4f}")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Plot the training history"""
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        
        # Save normalization parameters
        if self.param_stats is not None:
            np.save(os.path.join(os.path.dirname(model_path), 'param_stats.npy'), self.param_stats)
    
    def load_model(self, model_path, param_stats_path=None):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        # Load normalization parameters
        if param_stats_path is None:
            param_stats_path = os.path.join(os.path.dirname(model_path), 'param_stats.npy')
        
        if os.path.exists(param_stats_path):
            self.param_stats = np.load(param_stats_path, allow_pickle=True).item()
        
        return self.model

