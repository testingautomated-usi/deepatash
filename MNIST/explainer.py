
import tensorflow as tf
from config import EXPECTED_LABEL, MODEL, NUM_CLASSES
from skimage.color import rgb2gray
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D

from alibi.explainers import CEM
from alibi.explainers import IntegratedGradients

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
shape = (1,) + x_train.shape[1:]  # instance shape
kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes
            # in order for the first loss term to be minimized
beta = .1  # weight of the L1 loss term
gamma = 100  # weight of the optional auto-encoder loss term
c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or
              # the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = 10  # nb of updates for c
max_iterations = 1000  # nb of iterations per value of c
feature_range = (x_train.min(),x_train.max())  # feature range for the perturbed instance
clip = (-1000.,1000.)  # gradient clipping
lr = 1e-2  # initial learning rate
no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                  # perturbations towards this value means removing features, and away means adding features
                  # for our MNIST images, the background (-0.5) is the least informative,
                  # so positive/negative perturbations imply adding/removing features


model = tf.keras.models.load_model(MODEL)
ae = tf.keras.models.load_model('models/mnist_ae.h5')

# Initialize IntegratedGradients instance
n_steps = 50
method = "gausslegendre"
ig  = IntegratedGradients(model,
                          n_steps=n_steps,
                          method=method)


def explain_integrated_gradiant(X):
    predictions = model(X).numpy().argmax(axis=1)
    explanation = ig.explain([X],
                            baselines=None,
                            target=predictions)

    attrs = explanation.attributions[0]
    
    attr = attrs[0]
    # positive attributions
    attr_pos = attr.clip(0, 1)
    im_pos = attr_pos.squeeze()   

    return im_pos    



def explain_cem(X):
    # initialize CEM explainer and explain instance
    cem = CEM(model, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
            gamma=gamma, ae_model=ae, max_iterations=max_iterations,
            c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

    explanation = cem.explain(X)
    return explanation

def ae_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

if __name__ == "__main__":

    ae = ae_model()
    ae.summary()
    ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
    ae.save('models/mnist_ae.h5')





