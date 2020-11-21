import json
import requests

with open("tldr-legal.ndjson", "w+") as f:
    i = 0
    while True:
        url = f"https://tldrlegal.com/api/license?page={i}&pageSize=100&sort=-views"
        output = requests.get(url).json()
        if not output or len(output) == 0:
            print(output)
            break

        f.writelines((f"{(json.dumps(line))}\n" for line in output))
        i += 1
