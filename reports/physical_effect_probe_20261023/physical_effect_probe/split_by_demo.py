def splits(demos):
 demos=sorted(demos,key=lambda s:int(s.split('_')[-1]));assert len(demos)==10
 out=[]
 for f in range(5):
  test=demos[2*f:2*f+2];left=[x for x in demos if x not in test]
  out.append(dict(fold=f,train=left[1:],validation=left[:1],test=test))
 assert sorted(x for r in out for x in r['test'])==sorted(demos)
 return out
