import pickle

# Load extracted features
with open("features/features.pkl", "rb") as file:
    features = pickle.load(file)

print("Features loaded successfully!")

print("Total Images:", len(features))

# Display first image ID
first_image = list(features.keys())[0]

print("\nFirst Image ID:")
print(first_image)

print("\nFeature Shape:")
print(features[first_image].shape) 
