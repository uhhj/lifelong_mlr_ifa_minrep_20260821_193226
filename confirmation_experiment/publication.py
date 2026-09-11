"""Publish handoff to the already authenticated Windows host; no credentials copied."""
import argparse
import json
from confirmation_experiment.common import ROOT,RUNS,REPORTS,read,write,revision

def main():
    p=argparse.ArgumentParser();p.add_argument('--ack-sha');args=p.parse_args()
    if args.ack_sha:
        ready=read(REPORTS/'PUBLICATION_READY.json')
        assert args.ack_sha==ready['results_sha']==revision()
        write(REPORTS/'PUBLISHED.json',{'status':'PASS','results_sha':args.ack_sha,
              'evidence':'Authenticated local relay pushed and verified origin branch ref before acknowledging.'})
        return
    if (REPORTS/'PUBLISHED.json').exists():value=dict(phase='PUBLISHED',**read(REPORTS/'PUBLISHED.json'))
    elif (REPORTS/'PUBLICATION_READY.json').exists():value=dict(phase='READY',**read(REPORTS/'PUBLICATION_READY.json'))
    elif (RUNS/'FAILED.json').exists():value=dict(phase='FAILED',**read(RUNS/'FAILED.json'))
    elif (RUNS/'COMPLETE.json').exists():value=dict(phase='COMPLETE',**read(RUNS/'COMPLETE.json'))
    else:value={'phase':'WAITING'}
    print(json.dumps(value))

if __name__=='__main__':main()
