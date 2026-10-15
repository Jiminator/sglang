import hashlib,json,pathlib,sys
root=pathlib.Path('/sgl-workspace/sglang')
for relative,digest in json.loads(pathlib.Path(sys.argv[1]).read_text()).items():
 p=root/relative
 assert p.resolve().is_relative_to(root)
 assert (not p.exists()) if digest is None else hashlib.sha256(p.read_bytes()).hexdigest()==digest,relative
