class KNN():
    
    def __init__(self, k:int=4, p:int=2, type:str='classification'):
        
        """
        Initialize the KNN model

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors to consider. Defaults to 4.
        p : int or str, optional
            Distance metric to use. Can be either int or str. If int, it is
            interpreted as the power to use for the Minkowski metric. If str, it
            must be one of the following: 'euclidean', 'manhattan'.
            Defaults to 2, which corresponds to Euclidean distance.
        type : str, optional
            Type of problem to solve. Can be either 'classification' or
            'regression'. Defaults to 'classification'.
        """
        self.k = k
        self.p = p
        self.type = type
        
    def fit(self, X, y):
        """
        Fit the model to the training data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Labels for the training data.
        """

        self.X = X
        self.y = y
        
    def calcDistance(self, x1, x2):
        
        """
        Calculate the distance between two points

        Parameters
        ----------
        x1 : array-like of shape (n_features,)
            First point
        x2 : array-like of shape (n_features,)
            Second point

        Returns
        -------
        distancia : float
            Distance between the two points
        """
        if isinstance(x1, (list, tuple)):
            if self.p == 'euclidean':
                distancia = sum((x - y) ** 2 for x, y in zip(x1, x2)) ** 0.5
            elif self.p == 'manhattan':
                distancia = sum(abs(x - y) for x, y in zip(x1, x2))
            elif isinstance(self.p, str):
                raise ValueError(f'{self.p} is not a valid metric')
            elif self.p < 1:
                raise ValueError('p must be greater or equal to 1')
            else:
                distancia = sum(abs(x - y) ** self.p for x, y in zip(x1, x2)) ** (1 / self.p)
        else:
            distancia = ((x1 - x2) ** 2) ** 0.5
        return distancia
    
    def predict(self, X:list):
        """
        Predict the labels for the given data.

        Parameters
        ----------
        X : list of array-like of shape (n_features,)
            Data to predict the labels

        Returns
        -------
        predictions : list of labels
            Predicted labels
        """
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        """
        Predict the label for the given data.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Data to predict the label

        Returns
        -------
        prediction : label
            Predicted label
        """
        predictions = [self.calcDistance(x, X_train) for X_train in self.X]
        
        indices = self._argsort([sum(dist) for dist in predictions])[: self.k]
        
        if self.type in ('regression', 'r'):
            predictions = sum([self.y[i] for i in indices]) / len(indices)
    
        elif self.type in ('classification', 'c'):
            labels = [self.y[i] for i in indices]
            predictions = max(set(labels), key=labels.count)
            
        else:
            raise ValueError(f'{self.type} is not a valid type')
            
        return predictions
    
    def _argsort(self, X):
        """
        Sort the indices of the given list X by its values

        Parameters
        ----------
        X : list
            List of values to sort

        Returns
        -------
        list
            List of sorted indices
        """
        return [i[0] for i in sorted(enumerate(X), key=lambda x: x[1])]
    
class KNNUnidimensional():
    def __init__(self, k=4):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.y = y   
    def calcDistance(self, x1, x2): 
        distancia = ((x1 - x2) ** 2) ** 0.5
        return distancia
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        predictions = [self.calcDistance(x, X_train) for X_train in self.X]
        
        indices = self._argsort(predictions)[: self.k]
        labels = [self.y[i] for i in indices]
        
        predictions = max(set(labels), key=labels.count)
        return predictions
    
    def _argsort(self, X):
        return [i[0] for i in sorted(enumerate(X), key=lambda x: x[1])]