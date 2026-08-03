# Read the captions file

with open("captions.txt", "r", encoding="utf-8") as file:
    captions = file.readlines()

print("Total lines in captions file:", len(captions))

print("\nFirst 5 lines:\n")

for line in captions[:5]:
    print(line.strip())