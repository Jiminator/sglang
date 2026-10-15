import hashlib,json,os,pathlib,shutil
inputs=json.loads(pathlib.Path('/tmp/k3-release/release-inputs.json').read_text())
for i,a in enumerate(inputs['mooncake']['artifacts']):
 src=pathlib.Path('/tmp/k3-native')/str(i)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==a['sha256']
 dst=pathlib.Path(a['install_path'])
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(src,dst)
 os.chmod(dst,int(a['mode'],8))
