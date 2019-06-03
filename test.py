import numpy as np

from adfwk.models.ae import AutoEncoder
from adfwk.models.iforest import IsolationForest
from adfwk.models.vae import VariationalAutoEncoder


ae = AutoEncoder(hidden_neurons=[2, 1, 1, 2], verbose=0)

ae.fit(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]))

test = np.array([[1, 0, 1]])
print(ae.predict(test))
print(ae.predict_proba(test))
print(ae.decision_function(test))


vae = VariationalAutoEncoder(intermediate_dim=2, latent_dim=1, verbose=0)

vae.fit(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]))

test = np.array([[1, 0, 1]])
print(vae.predict(test))
print(vae.predict_proba(test))
print(vae.decision_function(test))


iforest = IsolationForest(verbose=0)

iforest.fit(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]))

test = np.array([[1, 0, 1]])
print(iforest.predict(test))
print(iforest.predict_proba(test))
print(iforest.decision_function(test))