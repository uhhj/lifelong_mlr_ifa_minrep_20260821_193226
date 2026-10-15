import multiprocessing,os
def check():
 import libero.libero.envs
 import coreset_exp.evaluate,coreset_exp.eval_metric
 from coreset_exp.common import S
 assert str(S)==os.environ['CORESET_SOURCE']
if __name__=='__main__':
 c=multiprocessing.get_context('spawn');p=c.Process(target=check);p.start();p.join(45)
 assert not p.is_alive() and p.exitcode==0
 print('PURE_SPAWN_IMPORT_PASS; environment episodes=0')
