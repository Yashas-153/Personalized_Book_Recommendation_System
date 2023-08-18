import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import pickle 

from preprocessing import pivot_table

nmf_model = NMF(n_components = 306, max_iter = 200,verbose = 2)

W = nmf_model.fit_transform(pivot_table)
pickle.dump(nmf_model,open("Models/nmf_model.pkl","wb"))


