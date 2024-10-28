import numpy as np
import pandas as pd
from typing import Callable, Tuple





class MultilayerPerceptron:
    def __init__(
        self,
        func: Callable = np.ReLu,
        number_hidden_layers: int = 2,
        number_units_in_hidden_layers: Tuple[int, int] = (64, 64),
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        self.func = func
        self.number_hidden_layers = number_hidden_layers
        self.number_units_in_hidden_layers = number_units_in_hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        
        pass
    
    def predict(self, X):
        
        pass
    
    def evaluate_acc(self, y, yh):
        
        pass
    