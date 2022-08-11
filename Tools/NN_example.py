from NN.ffnn import FFNN
from NN.scalar import Scalar
import numpy as np

# read in model and scalars
model = FFNN(0,0,model='NN/model')
X_scalar = Scalar()
X_scalar.load('NN/model/X_scalar.npy')
y_scalar = Scalar()
y_scalar.load('NN/model/y_scalar.npy')

# make a prediction
data = np.random.randn(4, 86) # spectra, this particular example is 4 different spectra, with 86 points in each spectra
# each spectra MUST have 86 points, or eles the model won't work. Linearly interpolate. 
X = X_scalar.transform(data) # transform the input
y = model.predict(X) # predict the value
y = y_scalar.untransform(y) # untransform the predicted value

print('predicted amp and std')
print(y)
