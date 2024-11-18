from abc import ABC, abstractmethod
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from enum import Enum
import numpy as np
from scripts.filter import filter_date_range

def run_model_analysis(models, train_data, mode, time_interval, val_data=None):

    # Filter data as before
    train_data = time_interval.filter(train_data)
    val_data = time_interval.filter(val_data) if val_data is not None else None
    
    results = {}
    for name, model in models.items():
        
        # Store results
        results[name] = model.fit_and_evaluate(train_data=train_data, val_data=val_data, mode=mode)
    
    return results

class ModelParameters(Enum):
    TEMPERATURE = 'Temperature (°F)'
    PRECIPITATION = 'Precipitation (in)'
    HUMIDITY = 'Relative Humidity (%)'
    CLOUD_COVER = 'Cloud Cover (%)'
    PRESSURE = 'Pressure (inHg)'

class Model(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, features=None):
        self.features = features or [
            ModelParameters.TEMPERATURE,
            ModelParameters.PRECIPITATION,
            ModelParameters.HUMIDITY,
            ModelParameters.CLOUD_COVER,
            ModelParameters.PRESSURE
        ]
        self.model = None
        self.scaler = None
    
    def prepare_data(self, data, mode, add_constant=False):
        """Prepare X and y data for modeling."""
        y = data[f'{mode}_residual']
        features = [feature.value for feature in self.features]
        X = data[features]
        
        if add_constant:
            X = sm.add_constant(X)
        
        return X, y
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions using the fitted model."""
        pass
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance with common metrics."""
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "R-squared": r2_score(y_true, y_pred)
        }
    
    def fit_and_evaluate(self, train_data, val_data=None, mode='subway'):
        """Fit model on training data and evaluate on both train and validation sets."""
        # Prepare training data
        X_train, y_train = self.prepare_data(train_data, mode)
        
        # Fit model on training data
        self.fit(X_train, y_train)
        
        # Make predictions and evaluate on training set
        y_train_pred = self.predict(X_train)
        train_metrics = self.evaluate(y_train, y_train_pred)

        residual_name = f'{mode}_residual'
        
        res = {
            'model': self,
            'train_metrics': train_metrics,
            'train_residual': train_data[residual_name],
            'summary': self.summary
        }
        
        # If validation data provided, evaluate on that too
        if val_data is not None:
            X_val, y_val = self.prepare_data(val_data, mode)
            y_val_pred = self.predict(X_val)
            val_metrics = self.evaluate(y_val, y_val_pred)
            res['val_metrics'] = val_metrics
            res['val_residual'] = val_data[residual_name]
        return res
    
    @property
    @abstractmethod
    def summary(self):
        """Return model summary or feature importance."""
        pass

class GLMModel(Model):
    """Gaussian GLM implementation."""
    
    def fit(self, X, y):
        self.model = sm.GLM(y, X, family=sm.families.Gaussian())
        self.results = self.model.fit()
        return self
    
    def predict(self, X):
        return self.results.predict(X)
    
    @property
    def summary(self):
        return self.results.summary().tables[1]

class QuantileModel(Model):
    """Quantile Regression implementation."""
    
    def __init__(self, features=None, quantile=0.5):
        super().__init__(features)
        self.quantile = quantile
    
    def fit(self, X, y):
        self.model = QuantReg(y, X)
        self.results = self.model.fit(q=self.quantile)
        return self
    
    def predict(self, X):
        return self.results.predict(X)
    
    @property
    def summary(self):
        return self.results.summary().tables[1]

class RobustModel(Model):
    """Robust Regression implementation."""
    
    def fit(self, X, y):
        self.model = RLM(y, X, M=sm.robust.norms.HuberT())
        self.results = self.model.fit()
        return self
    
    def predict(self, X):
        return self.results.predict(X)
    
    @property
    def summary(self):
        return self.results.summary().tables[1]

class LinearModel(Model):
    """Linear Regression implementation."""
    
    def __init__(self, features=None):
        super().__init__(features)
        self.scaler = StandardScaler()
        self.model = LinearRegression()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @property
    def summary(self):
        return dict(zip(self.features, self.model.coef_[1:]))

class GradientBoostingModel(Model):
    """Gradient Boosting implementation."""
    
    def __init__(self, features=None, **kwargs):
        super().__init__(features)
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            loss='absolute_error',
            random_state=42,
            **kwargs
        )
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @property
    def summary(self):
        return dict(zip(self.features, self.model.feature_importances_))

class XGBoostModel(Model):
    """XGBoost implementation."""
    
    def __init__(self, features=None, **kwargs):
        super().__init__(features)
        self.scaler = StandardScaler()
        # Default parameters that work well for many regression tasks
        default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 10,
            'random_state': 42
        }
        # Override defaults with any provided kwargs
        default_params.update(kwargs)
        self.model = xgb.XGBRegressor(**default_params)
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @property
    def summary(self):
        return dict(zip(self.features, self.model.feature_importances_))

class NaiveModel(Model):
    """Simple model that predicts a constant value that minimizes MSE on training data."""
    
    def __init__(self, features=None):
        super().__init__(features)
        self.optimal_value = None
    
    def fit(self, X, y):
        # For MSE loss, the optimal constant prediction is the mean
        # This minimizes sum((y - c)^2) where c is our constant prediction
        self.optimal_value = y.mean()
        return self
    
    def predict(self, X):
        return np.full(len(X), self.optimal_value)
    
    @property 
    def summary(self):
        return {"optimal_value": self.optimal_value}

