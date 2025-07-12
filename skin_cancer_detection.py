import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('model_skin_cancer.h5')

# Define class labels (skin cancer types)
classes = {
    0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign lichenoid keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevus',
    5: 'Pyogenic granulomas',
    6: 'Melanoma'
}
