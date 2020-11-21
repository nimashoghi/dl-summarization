import json
import requests


def get_licenses():
    i = 0
    while True:
        url = f"https://tldrlegal.com/api/license?page={i}&pageSize=100&sort=-views"
        output = requests.get(url).json()
        if not output or len(output) == 0:
            break
        i += 1

        yield from output


def get_license_info(id: str):
    url = f"https://tldrlegal.com/api/license/{id}"
    return requests.get(url).json()


def get_all_license_info():
    for license in get_licenses():
        info = get_license_info(license["id"])
        print(f"Got {license['title']}")
        yield info


with open("tldr-legal-info.ndjson", "w+") as f:
    for license_info in get_all_license_info():
        f.write(json.dumps(license_info) + "\n")
