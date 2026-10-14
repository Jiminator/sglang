import hashlib,json,pathlib,sys
for path,expected in json.loads(pathlib.Path(sys.argv[1]).read_text()).items():
    target=pathlib.Path('/sgl-workspace/sglang')/path
    if expected is None:
        assert not target.exists(), path
    else:
        assert hashlib.sha256(target.read_bytes()).hexdigest()==expected, path
print('Eight source paths verified')
