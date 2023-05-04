
import tensorflow as tf
from config import MODEL
from alibi.explainers import IntegratedGradients
import numpy as np
from config import EXPECTED_LABEL

model = tf.keras.models.load_model(MODEL)


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


def explain_integrated_gradiant_batch(X):
    predictions = np.full(X.shape[0], EXPECTED_LABEL)
    # predictions = tf.keras.utils.to_categorical(predictions, 10)
    explanation = ig.explain(X,
                            baselines=None,                            
                            target=predictions)

    attrs = explanation.attributions[0]
    im_poses = []
    for attr in attrs:
        # positive attributions
        attr_pos = attr.clip(0, 1)
        im_pos = attr_pos.squeeze() 
        im_poses.append(im_pos)  

    return im_poses   




# from xplique.attributions import  IntegratedGradients


# # Initialize IntegratedGradients instance
# n_steps = 50
# ig  = IntegratedGradients(model,
#                           steps=n_steps)

  
# def explain_integrated_gradiant_batch(X):
#     predictions = model(X).numpy().argmax(axis=1)
#     predictions = tf.keras.utils.to_categorical(predictions, 10)
#     explanation = ig.explain(X,                            
#                             predictions)

#     im_poses = []
#     for attr in explanation:
#         # positive attributions
#         attr_pos = attr.numpy().clip(0, 1)
#         im_pos = attr_pos.squeeze() 
#         im_poses.append(im_pos)  

#     return im_poses   

