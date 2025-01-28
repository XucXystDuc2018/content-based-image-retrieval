import numpy as np
import tensorflow as tf
import cv2
import os
from sklearn.neighbors import NearestNeighbors

# Load a pre-trained ResNet model
resnet_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

# Define a function to preprocess an image and extract features
def extract_features(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

# Define a directory containing the database of images to search through
database_directory = "D:/CBIR_dataset/raw-img/cane"
features_file = "database_features.npy"
filenames_file = "database_filenames.npy"

# Check if the features file already exists
if os.path.exists(features_file) and os.path.exists(filenames_file):
    # Load features and filenames from saved files
    database_features = np.load(features_file)
    database_filenames = np.load(filenames_file, allow_pickle=True).tolist()
    print("Loaded cached features and filenames.")
else:
    # Extract features from all images in the database and cache them
    database_features = []
    database_filenames = []
    for filename in os.listdir(database_directory):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(database_directory, filename)
            features = extract_features(image_path)
            database_features.append(features.flatten())  # Flatten the features
            database_filenames.append(filename)
    # Convert the list of features to a NumPy array
    database_features = np.array(database_features)
    # Save features and filenames to files
    np.save(features_file, database_features)
    np.save(filenames_file, database_filenames)
    print("Extracted and saved features and filenames.")

# Load the query image
query_image_path = "D:/CBIR_dataset/test/selc1u0qivz41.jpg"
query_features = extract_features(query_image_path).flatten()

# Define the number of similar images to retrieve
top_n = 10

# Use Nearest Neighbors for faster search
nn = NearestNeighbors(n_neighbors=top_n, metric='euclidean')
nn.fit(database_features)

# Find the top N similar images
distances, indices = nn.kneighbors([query_features])

# Display the top N most similar images and their distances
query_image = cv2.imread(query_image_path)
for i, index in enumerate(indices[0]):
    similar_image_filename = database_filenames[index]
    similar_image_path = os.path.join(database_directory, similar_image_filename)
    similar_image = cv2.imread(similar_image_path)
    distance = distances[0][i]
    print(f"Similar Image {i + 1}: Distance - {distance}")
    cv2.imshow(f"Similar Image {i + 1}", similar_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
