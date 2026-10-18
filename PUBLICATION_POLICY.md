# Workspace publication rule

User authorization, 2026-10-18: after each experiment round and each progress check or read-only audit, publish the relevant code, report and lightweight evidence to `https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226`, branch `MLR`, using `publish_mlr_branch`.

- This supersedes older no-push statements for publication only; scientific protocols and frozen execution contracts remain unchanged.
- Never publish checkpoints, model weights, demonstrations, latent tensors, buffer payloads, credentials, private keys or unreviewed archives. Metadata, hashes and episode outcome records are allowed.
- Preserve unrelated user edits and old reports. Stage only the current task's reviewed files. Never force push or fabricate empty commits.
- Verify the destination and branch, review filenames/sizes/secrets, run relevant read-only recomputation, commit, push, and verify the remote commit. Record a local receipt. Report any failure truthfully.
- Label ongoing runs PARTIAL with an observation timestamp; publishing a progress report does not mean experimental acceptance.
- Do not add tests, training, rollouts or scientific changes merely to publish. Keep automation prompts consistent with this standing authorization.
