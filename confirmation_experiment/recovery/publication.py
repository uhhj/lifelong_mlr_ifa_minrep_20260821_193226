import json
from confirmation_experiment.common import REPORTS,read
from confirmation_experiment.recovery.common import WORK

if __name__=='__main__':
    if (REPORTS/'PUBLICATION_READY.json').exists():
        from confirmation_experiment.publication import main
        main()
    elif (WORK/'FAILED.json').exists():print(json.dumps(dict(phase='FAILED',**read(WORK/'FAILED.json'))))
    else:print(json.dumps({'phase':'WAITING'}))
