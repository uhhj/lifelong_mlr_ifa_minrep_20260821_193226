"""Register additive observers, then execute the untouched native entry point."""
if __name__=='__main__':
    import runpy
    from confirmation_experiment.common import ROOT
    import confirmation_experiment.instrumentation
    runpy.run_path(str(ROOT/'libero/lifelong/lifelong_adaptation.py'),run_name='__main__')
