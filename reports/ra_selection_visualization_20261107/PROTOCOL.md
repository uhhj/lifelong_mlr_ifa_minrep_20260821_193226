# Offline RA-Coreset selection visualization

Scope: existing OBJECT and GOAL RA-Coreset selections, seeds 41–43, Tasks 6–9. No training, policy forward pass, environment interaction, new selection, or modification of production artifacts.

The principal figure uses the user's trajectory-coverage alternative. Each suite illustrates seed 41 / Task 6 (the first seed and task; complete local window identities are available for both). This example is not selected by a coverage advantage. The left panel uses actual normalized target time and demonstration identity, rather than a learned embedding. It displays K-Center, protected random anchors, and the final RA set in the same coordinates. The right panel compares full-quota Random, K-Center and RA: selected candidate windows divided by all candidate windows within each demonstration. The anchor subset is not the full-quota Random comparator. Ambiguous demonstration identities remain unknown; their counts are disclosed separately, never assigned to a convenient trajectory.

All six runs / four tasks are included in a supplementary original-distance trade-off figure and CSV. Existing preflight and production reports must agree. Its mean and maximum nearest-center distances come from the original frozen visual-action selection metric, not a two-dimensional projection. These are archived scalar calculations; raw distances are not newly recomputed in this task. Random nearest-distance distributions cannot be reconstructed from these aggregates and will not be fabricated.

Audit: verify source hashes, feature-to-window hash bindings where available, full population and equal quotas, exact seeded Random draw / anchor prefix, all protected anchors in RA, replay every recorded exchange, and agreement between original K-Center radius and recorded pre-exchange radius. Recompute demonstration counts from window identities for the illustrated examples and match saved K-Center / RA counts.

Task 9 storage follows the last training stage and is not subsequently replayed in the four-stage experiment. Coverage diagnostics do not by themselves establish policy success or causal mechanism. Windows are correlated and not independent statistical replicates; no window-level confidence interval or significance claim is used.

Transport: one remote inventory attempt failed during SSH banner negotiation before authentication / command execution. The transport script was restored. Existing local evidence is sufficient for the trajectory-coverage option; no raw latent payload is exported or published.
