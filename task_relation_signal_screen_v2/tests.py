import unittest
import numpy as np
from .numeric import ridge,Transform,linear_error,mmd,conditional,avg

class NumericalTests(unittest.TestCase):
    def test_near_constant(self):
        rows=[dict(x=1+i*1e-12) for i in range(18)]
        reg={'x':dict(scale_floor=1e-8,constant_floor=1e-8,bound=None)}
        p,raw,meta=ridge(rows,[dict(x=1e8)],np.arange(18)/18,['x'],reg)
        self.assertTrue(meta['transform'][0]['zero']);self.assertLess(abs(raw[0]),1)
    def test_fold_only_transform(self):
        reg={'x':dict(scale_floor=1e-8,constant_floor=1e-8,bound=None)}
        tr=Transform(['x'],reg).fit([{'x':0},{'x':1},{'x':2}]);before=str(tr.spec)
        z,meta=tr.apply([{'x':1e100},{'x':None}]);self.assertEqual(before,str(tr.spec));self.assertLessEqual(abs(z).max(),5);self.assertEqual(z[1,0],0)
    def test_monotonic_detection(self):
        x=np.linspace(-1,1,100);rows=[dict(x=float(a)) for a in x]
        reg={'x':dict(scale_floor=1e-8,constant_floor=1e-8,bound=[-1,1])}
        p,_,_=ridge(rows[:80],rows[80:],x[:80],['x'],reg)
        self.assertGreater(np.corrcoef(p,x[80:])[0,1],.99)
    def test_ridge_matches_augmented_system(self):
        x=np.linspace(-1,1,10);rows=[{'x':v} for v in x];reg={'x':dict(scale_floor=1e-8,constant_floor=1e-8,bound=[-1,1])}
        _,raw,m=ridge(rows,rows,x,['x'],reg)
        z=(x-x.mean())/x.std();coef=np.linalg.lstsq(np.r_[z[:,None]/np.sqrt(10),[[1.]]],np.r_[z/np.sqrt(10),0],rcond=None)[0]
        np.testing.assert_allclose(raw,z*coef[0]*x.std()+x.mean())
    def test_readout_crossfit(self):
        rng=np.random.RandomState(2);x=rng.randn(40,3);y=x[:,:2];half=np.arange(40)%2;ids=np.arange(40)//4
        self.assertTrue(np.isfinite(linear_error(x,y,half,ids)).all())
    def test_mmd_same(self):
        x=np.random.RandomState(3).randn(20,3);h=np.arange(20)%2
        np.testing.assert_allclose(mmd(x,x,h,h),[0,0],atol=1e-12)
    def test_matching_no_coverage(self):
        rng=np.random.RandomState(1);x=rng.randn(20,3);a=rng.randn(20,7);h=np.arange(20)%2
        r=conditional(x,x+1e6,a,a,h,h,np.ones(7))
        self.assertTrue(all(v[0]==0 and v[1] is None and v[2] is None for v in r))
    def test_trajectory_equal(self):
        self.assertEqual(avg([0,0,0,10],['a','a','a','b']),5)
    def test_exact_five_percentage_point_band(self):
        from .report_dev import discrete_effect
        np.testing.assert_array_equal(discrete_effect([.2-.15,.15-.2,.1,-.1]),[0,0,1,-1])
    def test_text_panel_maximin(self):
        from .panel import maximin
        self.assertEqual(maximin(['b','a','c'],np.array([[0.,1.],[1.,0.],[-1.,0.]])),[1,2,0])
    def test_shared_reference_reconstruction(self):
        from .confirmation_statistics import reconstruct
        rows=[dict(seed=48,source=0,target=2),dict(seed=48,source=1,target=2)]
        success={(48,'reference2',2):[0,1,0,1],(48,'transfer0_2',2):[0,1,0,1],(48,'transfer1_2',2):[0,1,0,1],(48,'source0',0):[1]*4,(48,'transfer0_2',0):[0]*4,(48,'source1',1):[1]*4,(48,'transfer1_2',1):[0]*4}
        ix={(48,t):np.array([1,1,1,0]) for t in (0,1,2)}
        np.testing.assert_array_equal(reconstruct(rows,success,ix),[[0,1],[0,1]])
    def test_max_t_analytic(self):
        from .confirmation_statistics import max_t_intervals
        x=np.array([-1.,0.,1.]);r=max_t_intervals([0,0],np.column_stack([x,2*x]),confidence=1.)
        np.testing.assert_allclose(r['lower'],[-1,-2]);np.testing.assert_allclose(r['upper'],[1,2])
        zero=max_t_intervals([0],np.zeros((10,1)));self.assertEqual(zero['lower'],[0]);self.assertEqual(zero['zero_variance'],[True])
    def test_nested_selection_never_reads_held_task_outcomes(self):
        from unittest.mock import patch
        from .common import PAIRS,PACKAGES
        from .analyze_dev import nested_dynamic_cv
        rng=np.random.RandomState(20260926)
        fields=['b_target','b_source','b_progress','base_progress20']+list(dict.fromkeys(f for ff in PACKAGES.values() for f in ff))
        rows=[dict(seed=k,source=s,target=t,**dict(zip(fields,rng.randn(len(fields))))) for k,s,t in PAIRS]
        reg={f:dict(scale_floor=1e-8,constant_floor=1e-8,bound=None) for f in fields};y=rng.randn(36)
        changed=y.copy();changed[[i for i,r in enumerate(rows) if 6 in (r['source'],r['target'])]]+=100
        with patch('task_relation_signal_screen_v2.analyze_dev.read',return_value={'new_C0_candidates':['S02','S07','S08']}):
            a,_,am=nested_dynamic_cv(rows,y,reg,dict(S02=1.,S07=2.,S08=3.))
            b,_,bm=nested_dynamic_cv(rows,changed,reg,dict(S02=1.,S07=2.,S08=3.))
        ix=[i for i,r in enumerate(rows) if r['target']==6]
        np.testing.assert_array_equal(a[ix],b[ix]);self.assertEqual(am[0],bm[0])
    def test_frozen_confirmation_prediction(self):
        from unittest.mock import patch
        from .confirm_signals import predict
        reg={'a':dict(bound=None,scale_floor=1e-8,constant_floor=1e-8)}
        train=[{'a':-1.},{'a':0.},{'a':1.}]
        expected,raw,fit=ridge(train,[{'a':.25}],np.array([-.3,0.,.4]),['a'],reg)
        frozen=dict(candidates=[dict(package='S04',endpoint='TRANSFER_300',features=['a'],fit=fit)],baselines=[])
        with patch('task_relation_signal_screen_v2.confirm_signals.read',return_value=frozen):
            result=predict({'a':.25},0)
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['prediction'],expected[0]);self.assertEqual(result[0]['raw_prediction'],raw[0])
            self.assertEqual(predict({'a':.25},20),[])
    def test_development_pipeline_synthetic(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .common import PACKAGES,PAIRS,write,table,unit,read
        from . import analyze_dev
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dest=root/'new';old=root/'old';dest.mkdir();old.mkdir()
            keys=['b_target','b_source','b_progress','base_progress1','base_progress5','base_progress20']+list(dict.fromkeys(f for fields in PACKAGES.values() for f in fields))
            registry={f:dict(bound=None,scale_floor=1e-8,constant_floor=1e-8) for f in keys}
            write(dest/'SIGNAL_REGISTRY.json',dict(features=registry,new_C0_candidates=['S02','S03','S05','S06','S07','S08','S09','S10','S11']))
            rng=np.random.RandomState(4);labels=[]
            for k,s,t in PAIRS:
                scores=dict(zip(keys,rng.randn(len(keys)).tolist()))
                dynamic=set(PACKAGES['S12']+PACKAGES['S13']+PACKAGES['S14']+['base_progress1','base_progress5','base_progress20'])
                a={f:v for f,v in scores.items() if f not in dynamic};b={f:v for f,v in scores.items() if f in dynamic}
                write(dest/'static'/unit(k,s,t)/'SIGNALS.json',dict(scores=a,halves=[a,a]))
                write(dest/'probes'/unit(k,s,t)/'MEASUREMENTS.json',dict(scores=b,halves=[b,b]))
                write(dest/'static'/unit(k,s,t)/'COMPUTE_COST.json',dict(version=2,package_seconds={p:1. for p in PACKAGES},scope='synthetic test only'))
                table(dest/'probes'/unit(k,s,t)/'PROBE_STEPS.csv',[dict(call=1,lr=0)])
                labels.append(dict(seed=k,source=s,target=t,TRANSFER_300=.4*np.tanh(scores['linear_probe']),DAMAGE_300=.3*np.tanh(scores['affinity'])))
            table(old/'TARGETS.csv',labels)
            with patch.object(analyze_dev,'OUT',dest),patch.object(analyze_dev,'OLD',old):analyze_dev.main()
            final=read(dest/'FINALISTS_FROZEN.json');self.assertLessEqual(len(final['candidates']),6)
            self.assertTrue(final['no_independent_confirmation_yet'])
            self.assertTrue((dest/'DEV_OOF_PREDICTIONS.csv').exists())

if __name__=='__main__':unittest.main()
