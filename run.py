import os
import train
import classify

if os.path.exists("./training"):
    print("Found training folder")
else:
    raise FileNotFoundError("Need a directory named \"training\" in the same directory as \"run.py\"")

if os.path.exists("./training/train.txt"):
    print("Found train.txt")
else:
    raise FileNotFoundError("Need a file named ""train.txt"" in the directory \"training\"")

if os.path.exists("./classification"):
    print("Found classification folder")
else:
    raise FileNotFoundError("Need a directory named \"classification\" in the same directory as \"run.py\"")

# train.train()
#classify.classify({0: 'Bark', 1: 'Flowers', 2: 'Sprinkles', 3: 'Metal', 4: 'Sand', 5: 'Water', 6: 'Nuts'})
