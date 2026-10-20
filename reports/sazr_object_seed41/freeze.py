from sazr_exp.common import *
assert js(O/"ADMISSION.json")["status"]=="PASS" and js(O/"PREFLIGHT.json")["status"]=="PASS"
assert js(O/"NOISE_INTERFACE.json")["status"]=="PASS"
assert not (O/"CONTRACT.json").exists()
files=js(O/"INPUT_SHA256.json")
for p in list((W/"source").rglob("*.py"))+[W/n for n in ("PROTOCOL.md","USER_PROTOCOL.txt","pipeline.py","supervisor.py","freeze.py","audit.py","env.sh")]:
    files[str(p)]=sha(p)
for name in ("PREFLIGHT.json","ADMISSION.json","BINDING.json","NOISE_INTERFACE.json"):files[str(O/name)]=sha(O/name)
write(O/"CONTRACT.json",dict(status="FROZEN",files=files,budget=dict(anchor=10,validation=20,formal=20,epochs=10,steps=1570),
    frozen=time.time(),scientific_changes="SAZR as declared in PROTOCOL.md"))
print("FROZEN",len(files),"files")
