import json


def load_json(path):
    with open(path) as json_file:
        o_file = json_file.read()
    return json.loads(o_file)
