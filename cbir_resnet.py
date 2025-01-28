import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import cv2
import shutil

# Load ResNet model
resnet_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

# Preload database features
database_directory = "D:/CBIR_dataset/raw-img/cane"
features_file = "database_features.npy"
filenames_file = "database_filenames.npy"

if os.path.exists(features_file) and os.path.exists(filenames_file):
    database_features = np.load(features_file)
    database_filenames = np.load(filenames_file, allow_pickle=True).tolist()
else:
    raise FileNotFoundError("Database features not found. Please preprocess the dataset.")

def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

def find_similar_images(query_image_path, result_folder, top_n=10):
    # Extract features for the query image
    query_features = extract_features(query_image_path)

    # Nearest Neighbors search
    nn = NearestNeighbors(n_neighbors=top_n, metric='euclidean')
    nn.fit(database_features)
    distances, indices = nn.kneighbors([query_features])

    # Prepare results
    results = []
    shutil.rmtree(result_folder)  # Clear old results
    os.makedirs(result_folder, exist_ok=True)
    for i, index in enumerate(indices[0]):
        similar_image_filename = database_filenames[index]
        similar_image_path = os.path.join(database_directory, similar_image_filename)
        result_image_path = os.path.join(result_folder, f"result_{i+1}.jpeg")
        shutil.copy(similar_image_path, result_image_path)
        results.append({
            'image_url': f'results/result_{i+1}.jpeg',
            'distance': distances[0][i]
        })
    return results
