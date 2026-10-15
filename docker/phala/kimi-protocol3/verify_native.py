import hashlib,json,pathlib
for path,digest in json.loads(pathlib.Path('/tmp/k3-release/native-files.json').read_text()).items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest,path
