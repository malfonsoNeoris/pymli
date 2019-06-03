from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer as SKLabelBinarizer


class LabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = SKLabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)
