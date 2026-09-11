from confirmation_experiment.recovery.common import install_abort_routing

if __name__=='__main__':
    install_abort_routing()
    import runpy
    runpy.run_module('confirmation_experiment.train_entry',run_name='__main__')
