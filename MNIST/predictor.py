# For Python 3.6 we use the base keras
import tensorflow as tf
import numpy as np

# local
from config import MODEL, EXPECTED_LABEL, NUM_CLASSES


class Predictor:

    @staticmethod
    def predict(img, model):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = tf.keras.utils.to_categorical(explabel, NUM_CLASSES)
        explabel = np.argmax(explabel.squeeze())

         #Predictions vector
        predictions = model.predict(img)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != EXPECTED_LABEL:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence

    def predict_batch(imgs, model):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = tf.keras.utils.to_categorical(explabel, NUM_CLASSES)
        explabel = np.argmax(explabel.squeeze())

        # Predictions vector
        batch_preds = model.predict(imgs)

        pred_conf = []

        for predictions in batch_preds:
            prediction1, prediction2 = np.argsort(-predictions)[:2]

            # Activation level corresponding to the expected class
            confidence_expclass = predictions[explabel]

            if prediction1 != EXPECTED_LABEL:
                confidence_notclass = predictions[prediction1]
            else:
                confidence_notclass = predictions[prediction2]

            confidence = confidence_expclass - confidence_notclass

            pred_conf.append(( prediction1, confidence))
        
        return pred_conf