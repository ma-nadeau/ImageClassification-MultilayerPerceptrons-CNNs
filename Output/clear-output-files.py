import os

# Specify the folder path
folder_path = "/home/2023/mnadea52/Fall2024/COMP551/ImageClassification-MultilayerPerceptrons-CNNs/Output"

# Loop through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.out'):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Deleted {file_path}")
