from sklearn.decomposition import TruncatedSVD
import numpy as np

class SVDBad(object):

    def __init__(self) -> None:
        self.model_svd = None
        self.model_linear = None

    def fit(self, array_input:np.ndarray, array_output_true:np.ndarray):
        self.model_svd = TruncatedSVD(array_output_true.shape[1])
        self.model_svd.fit(array_input)
       
    def transform(self, array_input:np.ndarray):
        return self.model_svd.transform(array_input)

    def inverse_transform(self, array_output_pred:np.ndarray):
        return self.model_svd.inverse_transform(array_output_pred)