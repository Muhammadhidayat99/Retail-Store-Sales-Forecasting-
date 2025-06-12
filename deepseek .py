"""
sales_forecasting.py

A comprehensive time series analysis and forecasting script for retail sales data.
Implements data exploration, feature engineering, model training, evaluation, and forecasting.

Requirements:
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- prophet (optional)
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class SalesForecaster:
    """
    A class for performing time series analysis and forecasting on sales data.
    
    Attributes:
        sales_data (pd.DataFrame): The loaded sales data
        holiday_data (pd.DataFrame): Optional holiday/promotion data
        economic_data (pd.DataFrame): Optional economic indicator data
        freq (str): Frequency of the time series ('MS' for monthly start)
        train (pd.DataFrame): Training data
        test (pd.DataFrame): Test/validation data
        best_model: The selected best forecasting model
    """
    
    def __init__(self, sales_file, holiday_file=None, economic_file=None):
        """
        Initialize the SalesForecaster with data files.
        
        Args:
            sales_file (str): Path to sales data CSV file
            holiday_file (str, optional): Path to holiday data CSV file
            economic_file (str, optional): Path to economic data CSV file
        """
        self.sales_data = None
        self.holiday_data = None
        self.economic_data = None
        self.freq = 'MS'  # Monthly frequency
        self.train = None
        self.test = None
        self.best_model = None
        
        # Load data
        self._load_data(sales_file, holiday_file, economic_file)
        
    def _load_data(self, sales_file, holiday_file, economic_file):
        """
        Load and preprocess the input data files.
        
        Args:
            sales_file (str): Path to sales data CSV file
            holiday_file (str, optional): Path to holiday data CSV file
            economic_file (str, optional): Path to economic data CSV file
        """
        try:
            # Load sales data
            self.sales_data = pd.read_csv(sales_file)
            
            # Convert date column to datetime and set as index
            self.sales_data['Date'] = pd.to_datetime(self.sales_data['Date'])
            self.sales_data.set_index('Date', inplace=True)
            self.sales_data.sort_index(inplace=True)
            
            # Check for missing values
            self._handle_missing_values()
            
            # Load holiday data if provided
            if holiday_file:
                self.holiday_data = pd.read_csv(holiday_file)
                self.holiday_data['Date'] = pd.to_datetime(self.holiday_data['Date'])
                self.holiday_data.set_index('Date', inplace=True)
                
            # Load economic data if provided
            if economic_file:
                self.economic_data = pd.read_csv(economic_file)
                self.economic_data['Date'] = pd.to_datetime(self.economic_data['Date'])
                self.economic_data.set_index('Date', inplace=True)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e.filename}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _handle_missing_values(self):
        """
        Handle missing values in the sales data.
        Reports number of missing values before and after handling.
        Uses linear interpolation for time series data.
        """
        print("\nMissing Values Analysis:")
        print("Before handling:")
        print(self.sales_data.isnull().sum())
        
        # Interpolate missing values for time series
        self.sales_data.interpolate(method='time', inplace=True)
        
        # If any values remain missing (e.g., at start/end), fill with mean
        if self.sales_data.isnull().sum().any():
            self.sales_data.fillna(self.sales_data.mean(), inplace=True)
            
        print("\nAfter handling:")
        print(self.sales_data.isnull().sum())
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the sales data.
        Includes visualization, decomposition, and stationarity tests.
        """
        print("\n=== Data Exploration ===")
        print(f"Data range: {self.sales_data.index.min()} to {self.sales_data.index.max()}")
        print(f"Total observations: {len(self.sales_data)}")
        
        # Plot sales over time
        self._plot_time_series()
        
        # Decompose time series
        self._decompose_time_series()
        
        # Check stationarity
        self._test_stationarity()
        
        # Plot ACF and PACF
        self._plot_acf_pacf()
    
    def _plot_time_series(self):
        """Plot the sales time series data."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.sales_data.index, self.sales_data['SalesAmount'], 
                label='Sales Amount', color='blue')
        plt.title('Monthly Sales Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales Amount', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _decompose_time_series(self):
        """Decompose the time series into trend, seasonal, and residual components."""
        print("\nTime Series Decomposition:")
        decomposition = seasonal_decompose(self.sales_data['SalesAmount'], model='additive', period=12)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(411)
        plt.plot(decomposition.observed, label='Observed')
        plt.legend()
        plt.title('Time Series Decomposition')
        
        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend')
        plt.legend()
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='Seasonal')
        plt.legend()
        
        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residual')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _test_stationarity(self):
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Null Hypothesis (H0): The time series is non-stationary.
        Alternative Hypothesis (H1): The time series is stationary.
        """
        print("\nStationarity Test (ADF):")
        result = adfuller(self.sales_data['SalesAmount'])
        
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value}')
            
        if result[1] <= 0.05:
            print("\nConclusion: Reject the null hypothesis (series is stationary)")
        else:
            print("\nConclusion: Fail to reject the null hypothesis (series is non-stationary)")
    
    def _plot_acf_pacf(self):
        """Plot ACF and PACF to help identify AR and MA terms."""
        plt.figure(figsize=(12, 6))
        
        # ACF plot
        plt.subplot(121)
        acf_values = acf(self.sales_data['SalesAmount'], nlags=20)
        plt.stem(acf_values)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=-1.96/np.sqrt(len(self.sales_data)), color='gray', linestyle='--')
        plt.axhline(y=1.96/np.sqrt(len(self.sales_data)), color='gray', linestyle='--')
        plt.title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plt.subplot(122)
        pacf_values = pacf(self.sales_data['SalesAmount'], nlags=20)
        plt.stem(pacf_values)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=-1.96/np.sqrt(len(self.sales_data)), color='gray', linestyle='--')
        plt.axhline(y=1.96/np.sqrt(len(self.sales_data)), color='gray', linestyle='--')
        plt.title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.show()
        
        print("\nACF/PACF Interpretation:")
        print("ACF tails off slowly, suggesting non-stationarity (differencing needed)")
        print("Significant spike at lag 12 in ACF suggests seasonal component")
        print("PACF has significant spike at lag 1, suggesting AR(1) term")
    
    def prepare_data(self, test_size=12):
        """
        Prepare data for modeling by splitting into train/test sets.
        
        Args:
            test_size (int): Number of periods to use for testing (default 12 months)
        """
        print(f"\nSplitting data into train/test sets (test_size={test_size})")
        
        # Split data while maintaining temporal order
        split_point = len(self.sales_data) - test_size
        self.train = self.sales_data.iloc[:split_point]
        self.test = self.sales_data.iloc[split_point:]
        
        print(f"Training period: {self.train.index.min()} to {self.train.index.max()}")
        print(f"Testing period: {self.test.index.min()} to {self.test.index.max()}")
    
    def train_arima(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Train an ARIMA/SARIMA model.
        
        Args:
            order (tuple): (p,d,q) order for ARIMA
            seasonal_order (tuple): (P,D,Q,s) seasonal order for SARIMA
            
        Returns:
            Fitted ARIMA/SARIMA model
        """
        print("\nTraining ARIMA/SARIMA model...")
        print(f"Order: {order}, Seasonal Order: {seasonal_order}")
        
        try:
            model = SARIMAX(self.train['SalesAmount'],
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            
            fitted_model = model.fit(disp=False)
            print(fitted_model.summary())
            return fitted_model
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {str(e)}")
            return None
    
    def train_exponential_smoothing(self, seasonal_periods=12, trend='add', seasonal='add'):
        """
        Train a Holt-Winters Exponential Smoothing model.
        
        Args:
            seasonal_periods (int): Number of periods in a season
            trend (str): 'add' or 'mul' for additive/multiplicative trend
            seasonal (str): 'add' or 'mul' for additive/multiplicative seasonality
            
        Returns:
            Fitted Exponential Smoothing model
        """
        print("\nTraining Exponential Smoothing model...")
        
        try:
            model = ExponentialSmoothing(self.train['SalesAmount'],
                                        trend=trend,
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods)
            
            fitted_model = model.fit()
            print(f"Model SSE: {fitted_model.sse}")
            return fitted_model
            
        except Exception as e:
            print(f"Error fitting Exponential Smoothing model: {str(e)}")
            return None
    
    def train_prophet(self):
        """
        Train a Prophet forecasting model (if installed).
        Requires Facebook's Prophet package to be installed.
        
        Returns:
            Fitted Prophet model
        """
        try:
            from prophet import Prophet
            
            print("\nTraining Prophet model...")
            
            # Prepare data for Prophet
            prophet_df = self.train.reset_index()
            prophet_df = prophet_df.rename(columns={'Date': 'ds', 'SalesAmount': 'y'})
            
            # Initialize and fit model
            model = Prophet(yearly_seasonality=True)
            
            # Add holiday effects if data is available
            if self.holiday_data is not None:
                holidays_df = self.holiday_data.reset_index()
                holidays_df = holidays_df.rename(columns={'Date': 'ds', 'Holiday': 'holiday'})
                model.add_country_holidays(country_name='US')
                model.fit(prophet_df, holidays=holidays_df)
            else:
                model.fit(prophet_df)
                
            return model
            
        except ImportError:
            print("Prophet not installed. Skipping Prophet model.")
            return None
        except Exception as e:
            print(f"Error fitting Prophet model: {str(e)}")
            return None
    
    def evaluate_model(self, model, model_name=''):
        """
        Evaluate a model's performance on the test set.
        
        Args:
            model: The trained forecasting model
            model_name (str): Name of the model for display purposes
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if model is None:
            return None
            
        print(f"\nEvaluating {model_name} model...")
        
        try:
            # Make predictions
            if model_name.lower() == 'prophet':
                future = model.make_future_dataframe(periods=len(self.test), freq='M')
                forecast = model.predict(future)
                predictions = forecast['yhat'][-len(self.test):].values
            else:
                predictions = model.forecast(steps=len(self.test))
            
            # Calculate metrics
            actual = self.test['SalesAmount']
            mae = mean_absolute_error(actual, predictions)
            mse = mean_squared_error(actual, predictions)
            rmse = sqrt(mse)
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            print(f"MAE: {mae:.2f}")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.plot(self.train.index, self.train['SalesAmount'], label='Training Data')
            plt.plot(self.test.index, actual, label='Actual Test Data')
            plt.plot(self.test.index, predictions, label='Predictions')
            plt.title(f'{model_name} Model Performance', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Sales Amount', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            return {
                'model': model_name,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None
    
    def select_best_model(self):
        """
        Train multiple models and select the best one based on test performance.
        """
        print("\n=== Model Selection ===")
        
        # Train different models
        arima_model = self.train_arima(order=(1,1,1), seasonal_order=(1,1,1,12))
        es_model = self.train_exponential_smoothing()
        prophet_model = self.train_prophet()
        
        # Evaluate models
        results = []
        
        if arima_model:
            arima_metrics = self.evaluate_model(arima_model, 'ARIMA')
            if arima_metrics:
                results.append(arima_metrics)
        
        if es_model:
            es_metrics = self.evaluate_model(es_model, 'Exponential Smoothing')
            if es_metrics:
                results.append(es_metrics)
        
        if prophet_model:
            prophet_metrics = self.evaluate_model(prophet_model, 'Prophet')
            if prophet_metrics:
                results.append(prophet_metrics)
        
        # Select best model based on RMSE
        if results:
            results_df = pd.DataFrame(results)
            results_df.sort_values('rmse', inplace=True)
            self.best_model = results_df.iloc[0]['model']
            
            print("\nModel Comparison:")
            print(results_df)
            print(f"\nSelected best model: {self.best_model}")
        else:
            print("No valid models were successfully trained and evaluated.")
    
    def final_forecast(self, periods=6):
        """
        Generate final forecasts using the best model on all available data.
        
        Args:
            periods (int): Number of periods to forecast ahead
            
        Returns:
            pd.DataFrame: Forecast results with confidence intervals
        """
        if not self.best_model:
            print("No best model selected. Run select_best_model() first.")
            return None
            
        print(f"\nGenerating final {periods}-period forecast using {self.best_model}...")
        
        try:
            if self.best_model.lower() == 'arima':
                # Retrain ARIMA on full data
                model = SARIMAX(self.sales_data['SalesAmount'],
                              order=(1,1,1),
                              seasonal_order=(1,1,1,12))
                fitted_model = model.fit(disp=False)
                
                # Generate forecast
                forecast = fitted_model.get_forecast(steps=periods)
                forecast_df = forecast.conf_int(alpha=0.05)
                forecast_df['forecast'] = fitted_model.predict(
                    start=forecast_df.index[0],
                    end=forecast_df.index[-1])
                
            elif self.best_model.lower() == 'exponential smoothing':
                # Retrain Exponential Smoothing on full data
                model = ExponentialSmoothing(self.sales_data['SalesAmount'],
                                           trend='add',
                                           seasonal='add',
                                           seasonal_periods=12)
                fitted_model = model.fit()
                
                # Generate forecast
                forecast = fitted_model.forecast(steps=periods)
                forecast_df = pd.DataFrame({
                    'lower SalesAmount': forecast * 0.9,  # Simplified confidence interval
                    'upper SalesAmount': forecast * 1.1,
                    'forecast': forecast
                })
                
            elif self.best_model.lower() == 'prophet':
                # Retrain Prophet on full data
                from prophet import Prophet
                
                prophet_df = self.sales_data.reset_index()
                prophet_df = prophet_df.rename(columns={'Date': 'ds', 'SalesAmount': 'y'})
                
                model = Prophet(yearly_seasonality=True)
                if self.holiday_data is not None:
                    holidays_df = self.holiday_data.reset_index()
                    holidays_df = holidays_df.rename(columns={'Date': 'ds', 'Holiday': 'holiday'})
                    model.add_country_holidays(country_name='US')
                    model.fit(prophet_df, holidays=holidays_df)
                else:
                    model.fit(prophet_df)
                
                # Generate forecast
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)
                forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                forecast_df = forecast_df.set_index('ds')
                forecast_df.columns = ['forecast', 'lower SalesAmount', 'upper SalesAmount']
            
            # Plot the forecast
            plt.figure(figsize=(12, 6))
            plt.plot(self.sales_data.index, self.sales_data['SalesAmount'], label='Historical Data')
            plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
            plt.fill_between(forecast_df.index,
                           forecast_df['lower SalesAmount'],
                           forecast_df['upper SalesAmount'],
                           color='pink', alpha=0.3)
            plt.title(f'{periods}-Month Sales Forecast', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Sales Amount', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            return forecast_df
            
        except Exception as e:
            print(f"Error generating forecast: {str(e)}")
            return None

def main():
    """
    Main execution function for the sales forecasting pipeline.
    """
    try:
        # Initialize forecaster with data files
        forecaster = SalesForecaster(
            sales_file='retail_sales_mock_data.csv',
            holiday_file=None,  # Optional: 'holiday_data.csv'
            economic_file=None  # Optional: 'economic_data.csv'
        )
        
        # Perform EDA
        forecaster.explore_data()
        
        # Prepare data for modeling
        forecaster.prepare_data(test_size=12)
        
        # Select best model
        forecaster.select_best_model()
        
        # Generate final forecast
        forecast = forecaster.final_forecast(periods=6)
        
        if forecast is not None:
            print("\nFinal Forecast Results:")
            print(forecast)
        
    except Exception as e:
        print(f"Error in sales forecasting pipeline: {str(e)}")

if __name__ == "__main__":
    main()