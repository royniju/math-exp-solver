import pickle

# Replace this list with the exact order used during training
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']

# Create a mapping from index to character
idx_to_char = {i: char for i, char in enumerate(characters)}

# Save to file
with open("label_map.pkl", "wb") as f:
    pickle.dump(idx_to_char, f)

print("label_map.pkl saved successfully.")
