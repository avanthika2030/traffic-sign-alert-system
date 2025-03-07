import keras
from keras.models import load_model

# Load your .h5 model
model = load_model('traffic_classifier.h5')

# Extract the architecture (JSON) and weights
model_structure = model.to_json()  # Converts architecture to JSON format
model_weights = model.get_weights()  # Extracts the weights as NumPy arrays

import pickle

# Combine architecture and weights into a dictionary
model_data = {
    'architecture': model_structure,
    'weights': model_weights
}

# Save the model data to a pickle file
with open('traffic_classifier.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model saved to pickle file successfully!")
