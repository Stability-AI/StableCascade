import json
import subprocess
import yaml
import os
from .bucketeer import Bucketeer

class MultiFilter():
    def __init__(self, rules, default=False):
        self.rules = rules
        self.default = default

    def __call__(self, x):
        try:
            x_json = x['json']
            if isinstance(x_json, bytes):
                x_json = json.loads(x_json) 
            validations = []
            for k, r in self.rules.items():
                if isinstance(k, tuple):
                    v = r(*[x_json[kv] for kv in k])
                else:
                    v = r(x_json[k])
                validations.append(v)
            return all(validations)
        except Exception:
            return False

class MultiGetter():
    def __init__(self, rules):
        self.rules = rules

    def __call__(self, x_json):
        if isinstance(x_json, bytes):
            x_json = json.loads(x_json) 
        outputs = []
        for k, r in self.rules.items():
            if isinstance(k, tuple):
                v = r(*[x_json[kv] for kv in k])
            else:
                v = r(x_json[k])
            outputs.append(v)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

def setup_webdataset_path(paths, cache_path=None):
    if cache_path is None or not os.path.exists(cache_path):
        tar_paths = []
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if path.strip().endswith(".tar"):
                # Avoid looking up s3 if we already have a tar file
                tar_paths.append(path)
                continue
            bucket = "/".join(path.split("/")[:3])
            result = subprocess.run([f"aws s3 ls {path} --recursive | awk '{{print $4}}'"], stdout=subprocess.PIPE, shell=True, check=True)
            files = result.stdout.decode('utf-8').split()
            files = [f"{bucket}/{f}" for f in files if f.endswith(".tar")]
            tar_paths += files

        with open(cache_path, 'w', encoding='utf-8') as outfile:
            yaml.dump(tar_paths, outfile, default_flow_style=False)
    else:
        with open(cache_path, 'r', encoding='utf-8') as file:
            tar_paths = yaml.safe_load(file)

    tar_paths_str = ",".join([f"{p}" for p in tar_paths])
    return f"pipe:aws s3 cp {{ {tar_paths_str} }} -"
