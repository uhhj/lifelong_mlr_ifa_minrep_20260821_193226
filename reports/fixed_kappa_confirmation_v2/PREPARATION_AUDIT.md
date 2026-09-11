Preflight recovery record

No formal training or evaluation was started during preparation retries. The new seed audit preceded all new caches.

1. A BriefPolicy configuration was initially checked for front-end flags. Those flags belong to the CompletePolicy feature-generation configuration. Corrected before cache generation.
2. System NVIDIA userspace 580.178.04 did not match kernel 580.173.02. CUDA and official process-local GL/gpucomp packages were hash checked; actual EGL reports NVIDIA 580.173.02 on RTX 4090. No system driver was installed or changed.
3. Seed45 generator exited successfully. Subsequent validation compared its current official checkpoint path against the historical obsolete /data prefix. The current checkpoint hash matches the previously audited formal protocol. Seed45 cache was retained without regeneration; seeds46/47 were generated once each.
4. Initial seed42 smoke exposed a double-extension bug in newly added replay-origin recovery metadata. Corrected only in confirmation instrumentation, with a filename regression test. Its failed smoke and prior preflight artifacts remain separately preserved. No historical method source changed.
5. Final validation passed 22 new unit tests, 18 legacy unit tests, native real-policy kappa0/1 equality, native vs observer-off vs observer-on GPU equality, real trigger/control snapshot one-step restoration, and two-arm torchrun smoke. Maximum numerical difference was zero. Formal 100-epoch task6 and actual trial parity is a separate mandatory queue barrier.

The server has no GitHub credential. Git bundles are relayed to the authenticated Windows host for scoped branch pushes. The local machine must stay running/online for automatic final publication; server training and evaluation run independently.
