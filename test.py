import numpy as np

from adfwk.models.ae import AutoEncoder
from adfwk.models.iforest import IsolationForest
from adfwk.models.vae import VariationalAutoEncoder

train = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
test = np.array([[1, 0, 1]])

ae = AutoEncoder(hidden_neurons=[2, 1, 1, 2], verbose=0)
ae.fit(train)
print(ae.predict(test))
print(ae.predict_proba(test))
print(ae.decision_function(test))
ae.save('./data/models/ae')
ae = AutoEncoder.load('./data/models/ae')
print(ae.predict(test))
print(ae.predict_proba(test))
print(ae.decision_function(test))

vae = VariationalAutoEncoder(intermediate_dim=2, latent_dim=1, verbose=0)
vae.fit(train)
print(vae.predict(test))
print(vae.predict_proba(test))
print(vae.decision_function(test))
# TODO: arreglar el read del VAE. No esta tomando el K.random como un input
# vae.save('./data/models/vae')
# vae = VariationalAutoEncoder.load('./data/models/vae')
# print(vae.predict(test))
# print(vae.predict_proba(test))
# print(vae.decision_function(test))

iforest = IsolationForest(verbose=0)
iforest.fit(train)
print(iforest.predict(test))
print(iforest.predict_proba(test))
print(iforest.decision_function(test))
iforest.save('./data/models/iforest')
iforest = IsolationForest.load('./data/models/iforest')
print(iforest.predict(test))
print(iforest.predict_proba(test))
print(iforest.decision_function(test))
