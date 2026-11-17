
import pandas as pd

# Load the three dataset files
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
validation = pd.read_csv("dataset/validation.csv")

print("DATASET LOADED SUCCESSFULLY!")

print("\nTRAINING DATA:")
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())
print(train.head())

print("\nTESTING DATA:")
print("Shape:", test.shape)

print("\nVALIDATION DATA:")
print("Shape:", validation.shape)

print("\nDATA TYPES:")
print(train.dtypes)

print("\nMISSING VALUES:")
print(train.isnull().sum())

print("\nDATASET INFORMATION:")
print(train.info())
