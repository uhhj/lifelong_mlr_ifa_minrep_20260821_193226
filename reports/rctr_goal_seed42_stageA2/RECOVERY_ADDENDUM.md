# Dedicated-environment recovery, approved 2026-10-20

The first process exited1, with135 prefix steps,48 engineering steps and zero committed screening branches. First branch (initial20, repeat0, normal) began but its first env.step raised `AttributeError: 'MjSim' object has no attribute 'data'` while updating a robot observable. Some internal physics substeps may already have executed; this is one technical attempt, **not a task failure or a completed control step**.

Root cause: robosuite hard reset destroys the old MjSim, while saved observable sensor functions retain references to its robot/simulator. Immediate same-snapshot checks passed before later resets; they did not test the lifespan across collection of all three snapshots.

User explicitly approved:
- additional maximum135 prefix control steps and48 strict engineering control steps;
- same three states and same prefix noise, exact physical-state and native observation hashes must match the original collection;
- original333 unique planned continuations unchanged, one prior failed technical attempt separate, maximum334 total screening attempts;
- no added candidates, training or new initial states.

Frozen runner.py and its CONTRACT/exit/log remain unchanged. recovery_runner.py is a separately hashed lifecycle adapter, with complete diff in RECOVERY_DIFF.patch. Each snapshot owns a distinct live environment. All three environments are constructed and all snapshots collected before the repeated restoration checks; this exercises the lifespan that the first check missed. No simulator reset or destruction occurs between snapshot capture and its candidate branches.

Recovery outputs go to original output/engineering_recovery/dedicated_env_v1. Old hashes are rechecked before loading. Original state/observation/noise/initial identity comparisons are exact, before screening. If any check fails, stop and preserve failure; no automatic additional retries. The actual parent model, action/noise rules, candidate order, thresholds, success criteria and maximum horizon are unchanged.

Original failure remains a failure. Recovery success, if achieved, will have separate exit and acceptance evidence and must include original counts in total cost.
