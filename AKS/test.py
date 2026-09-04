# import pandas as pd
import json
path = 'outscores/nextqa/blip/video.json'
with open(path) as file:
    for line in file.readlines():
        data = json.loads(line)
        print(len(set(data)))
