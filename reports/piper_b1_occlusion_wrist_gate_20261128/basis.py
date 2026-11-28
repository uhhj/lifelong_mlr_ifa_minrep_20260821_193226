"""Shared archive paths and frozen model loading; never construct device interfaces."""
import sys,json,hashlib,importlib.util
from pathlib import Path
import cv2
import numpy as np
dest=Path(__file__).resolve().parent
work,old,prev,edge=map(Path,sys.argv[1:5])
sys.path[:0]=[str(edge),str(prev),str(old)]
from survey import rows
import replay as rp
read=rp.read
def dump(name,value):(dest/name).write_text(json.dumps(value,allow_nan=False,indent=1))
def model(folder,filename):
    sys.path.insert(0,str(work/'piper_test_console_pika_b1_full_v1'))
    spec=importlib.util.spec_from_file_location(filename,folder/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.GroundedODE(read(folder/'candidate.json'))
def setup():
    cv2.setNumThreads(1);rp.torch.set_num_threads(2)
    sys.path.insert(0,str(work/'offline_diagnostics/b1_vision_contrasts_20261124'))
    return model(edge,'ode_edge.py')
