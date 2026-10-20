"""Official immutable data acquisition with verified lossless HDF5 repacking.

Only this program's verified raw staging copies may be removed. Existing
experiment data, models and downloaded local originals are never touched.
"""
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import h5py
import numpy as np
import requests

Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
ROOT=Path("/data/mlr_ifa/datasets/libero50_acquired")
STAGING=Path("/data/mlr_ifa/datasets/libero50_download_staging")
ALTERNATE=Path("/root/mlr_ifa_datasets/libero50_acquired")


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w") as f:
        json.dump(value,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def download(item,manifest):
    raw=STAGING/item["path"];raw.parent.mkdir(parents=True,exist_ok=True)
    if raw.exists():
        assert raw.stat().st_size==item["size"] and sha(raw)==item["lfs"]["oid"]
        return raw
    partial=raw.with_suffix(".hdf5.partial")
    url="https://huggingface.co/datasets/%s/resolve/%s/%s?download=true"%(
        manifest["repo"],manifest["revision"],item["path"])
    failures=[]
    for attempt in range(1,9):
        try:
            start=partial.stat().st_size if partial.exists() else 0
            reserve=item["size"]-start+2*1024**3
            assert shutil.disk_usage(STAGING).free>reserve,"Raw staging reserve exhausted"
            if start<item["size"]:
                with requests.get(url,headers={"Range":"bytes=%d-"%start} if start else {},
                                  stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    append=start>0 and response.status_code==206
                    if response.status_code==206:
                        expected="bytes %d-"%(start if append else 0)
                        assert response.headers.get("Content-Range","").startswith(expected)
                    with partial.open("ab" if append else "wb") as f:
                        tick=time.monotonic()
                        for block in response.iter_content(4*1024**2):
                            if block:f.write(block)
                            if time.monotonic()-tick>20:
                                f.flush()
                                write(Q/"LIBERO50_SERVER_CURRENT.json",dict(status="DOWNLOADING",
                                    path=item["path"],bytes=partial.stat().st_size,total=item["size"],
                                    attempt=attempt,unix=time.time(),failures=failures))
                                tick=time.monotonic()
            assert partial.stat().st_size==item["size"],"Truncated upstream response"
            actual=sha(partial)
            if actual!=item["lfs"]["oid"]:
                partial.rename(partial.with_suffix(".badsha-%d"%time.time_ns()))
                raise ValueError("Upstream SHA mismatch")
            os.replace(partial,raw)
            return raw
        except Exception as exc:
            failures.append(dict(type=type(exc).__name__,attempt=attempt,unix=time.time()))
            write(Q/"LIBERO50_SERVER_CURRENT.json",dict(status="NETWORK_OR_STORAGE_RETRY",
                path=item["path"],failures=failures,unix=time.time()))
            time.sleep(min(5*attempt,40))
    raise RuntimeError("Acquisition failed after bounded retries: "+item["path"])


def same_attrs(a,b):
    assert set(a.attrs)==set(b.attrs)
    for key in a.attrs:
        assert np.array_equal(np.asarray(a.attrs[key]),np.asarray(b.attrs[key])),key


def repack(raw,target):
    temporary=target.with_suffix(".hdf5.repacking")
    if temporary.exists():
        temporary.rename(temporary.with_suffix(".interrupted-%d"%time.time_ns()))
    target.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256();datasets=0
    with h5py.File(raw,"r") as source,h5py.File(temporary,"w") as dest:
        for key,value in source.attrs.items():dest.attrs[key]=value
        def copy(name,obj):
            nonlocal datasets
            if isinstance(obj,h5py.Group):
                new=dest.require_group(name)
            else:
                assert not obj.dtype.hasobject,"Unsupported variable/object dataset; do not silently convert"
                opts={}
                if obj.shape and all(obj.shape) and obj.nbytes>1024:
                    chunk=(min(8,obj.shape[0]),)+obj.shape[1:]
                    opts=dict(chunks=chunk,compression="gzip",compression_opts=4,shuffle=True)
                new=dest.create_dataset(name,shape=obj.shape,dtype=obj.dtype,**opts)
                if obj.shape:
                    for start in range(0,len(obj),32):new[start:start+32]=obj[start:start+32]
                else:new[()]=obj[()]
                datasets+=1
            for key,value in obj.attrs.items():new.attrs[key]=value
        source.visititems(copy)
        dest.flush()
    with h5py.File(raw,"r") as source,h5py.File(temporary,"r") as dest:
        same_attrs(source,dest)
        oldnames=[];newnames=[]
        source.visit(oldnames.append);dest.visit(newnames.append)
        assert oldnames==newnames
        for name in oldnames:
            a,b=source[name],dest[name]
            assert type(a)==type(b)
            same_attrs(a,b)
            if isinstance(a,h5py.Dataset):
                assert a.shape==b.shape and a.dtype==b.dtype
                digest.update(name.encode()+b"\0"+str(a.dtype).encode()+str(a.shape).encode())
                if a.shape:
                    for start in range(0,len(a),32):
                        x,y=a[start:start+32],b[start:start+32]
                        assert x.tobytes()==y.tobytes(),name
                        digest.update(x.tobytes())
                else:
                    assert np.asarray(a[()]).tobytes()==np.asarray(b[()]).tobytes()
                    digest.update(np.asarray(a[()]).tobytes())
    os.replace(temporary,target)
    return dict(logical_datasets=datasets,logical_array_sha256=digest.hexdigest(),
                all_attributes_equal=True,all_dataset_bytes_equal=True,
                packed_sha256=sha(target),packed_bytes=target.stat().st_size)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);STAGING.mkdir(parents=True,exist_ok=True)
    ALTERNATE.mkdir(parents=True,exist_ok=True)
    lock=(Q/"libero50_acquisition.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(Q/"libero50_acquisition.pid.json",dict(pid=os.getpid(),unix=time.time()))
    manifest=json.loads((Q/"LIBERO50_DOWNLOAD_MANIFEST.json").read_text())
    assert len(manifest["files"])==50
    results=[]
    for index,item in enumerate(manifest["files"]):
        record=Q/"libero50_acquisition"/(Path(item["path"]).stem+".json")
        if record.exists():
            done=json.loads(record.read_text())
            assert sha(Path(done["packed_path"]))==done["packed_sha256"]
            results.append(done);continue
        # Alternate physical locations to preserve room for experiments on both disks.
        preferred=ROOT if index%2==0 else ALTERNATE
        other=ALTERNATE if preferred==ROOT else ROOT
        location=preferred if shutil.disk_usage(preferred).free>item["size"]+4*1024**3 else other
        assert shutil.disk_usage(location).free>item["size"]+4*1024**3,"Insufficient safe packing reserve"
        target=location/item["path"]
        assert not target.exists(),"Unexpected existing packed asset; explicit verification required"
        raw=download(item,manifest)
        write(Q/"LIBERO50_SERVER_CURRENT.json",dict(status="LOSSLESS_REPACK",path=item["path"],unix=time.time()))
        proof=repack(raw,target)
        logical=ROOT/item["path"]
        if logical!=target:
            logical.parent.mkdir(parents=True,exist_ok=True)
            assert not logical.exists();logical.symlink_to(target)
        done=dict(status="VERIFIED",path=item["path"],upstream_sha256=item["lfs"]["oid"],
            upstream_bytes=item["size"],revision=manifest["revision"],packed_path=str(target),
            logical_path=str(logical),unix=time.time(),**proof)
        write(record,done)
        # The only removal: our own SHA-verified raw staging copy, after complete
        # byte-for-byte array and attribute equality plus durable packed receipt.
        assert STAGING.resolve() in raw.resolve().parents
        assert sha(raw)==item["lfs"]["oid"] and sha(target)==proof["packed_sha256"]
        raw.unlink()
        results.append(done)
        write(Q/"LIBERO50_SERVER_STATUS.json",dict(status="ACQUIRING",verified=len(results),total=50,
            results=results,unix=time.time(),data_only=True,pretrained_checkpoint_available=False))
    write(Q/"LIBERO50_SERVER_STATUS.json",dict(status="DATA_VERIFIED",verified=50,total=50,
        results=results,unix=time.time(),data_only=True,pretrained_checkpoint_available=False))


if __name__=="__main__":
    main()
