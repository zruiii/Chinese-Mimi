import json
import numpy as np

res = []
for line in open("data/wenetspeech4tts_Standard_train.jsonl"):
    res.append(json.loads(line.strip())['duration'])

print(f"training samples: {len(res)}")
print(f"average duration: {np.mean(res)}")
print(f"min duration: {np.min(res)}")
print(f"max duration: {np.max(res)}")

sort_res = sorted(res)
print(f"1/5: {sort_res[int(0.2 * len(res))]}")
print(f"4/5: {sort_res[int(0.8 * len(res))]}")