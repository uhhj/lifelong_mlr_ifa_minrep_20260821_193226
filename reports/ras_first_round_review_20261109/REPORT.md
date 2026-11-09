# RAS first round editorial assessment

Observation date: 2026-11-09 Asia/Shanghai. Status: completed manuscript-content review; not an experimental acceptance audit. No training or rollout was performed. The supplied November 7 DOCX is the assessment boundary; later experiments are not assumed to be present in it.

## Judgment

For Robotics and Autonomous Systems, a subjective first-round probability of receiving a revision decision rather than desk rejection or rejection after review is approximately 40–60%, centered around 50%, conditional on completing an English submission with an abstract, references and consistent cross-references. This is an editorial judgment, not a measured journal rate, calibrated forecast, confidence interval or eventual acceptance probability. The supplied file is a Chinese working manuscript without an abstract or reference list and should not be submitted unchanged.

## Supporting evidence

The topic matches the journal's published scope covering robot control and learning and theoretical, computational and experimental aspects of autonomous systems. Official scope evidence: https://shop.elsevier.com/subjects/journals/physical-sciences-and-engineering/computer-science/artificial-intelligence/artificial-intelligence-expert-systems-and-knowledge-based-systems . Direct ScienceDirect scope and author-guide requests returned 403; no unverified detailed author requirements or journal acceptance rates are used.

The manuscript reports three seeds, two LIBERO suites, cumulative component ablations, a matched optimizer-call control, retention-weight sensitivity, gate behavior and task-stage comparisons. In Table 1, AUC gains over MLR+IFA are 18.2 percentage points on OBJECT and 10.9 on GOAL. Table 3 gains over matched-budget execution supervision are 5.6 and 3.6 points. Gate gains over ungated support-weighted retention are 1.9 and 3.8 points. These are arithmetic comparisons of reported means, not significance tests or independent raw-outcome verification.

## Main decision risks and priorities

1. Main-table comparability is not sufficiently documented. Identify which results are reproduced and which are cited, and state task split/order, architecture, demonstration and replay budgets, metric definitions, evaluation policy and interaction access. DEBR collects 10 training episodes per task and uses extra successful-history memory. Optimizer-call matching does not equalize those resources. This is a reporting gap, not proof the comparisons are invalid.
2. The title foregrounds support calibration, but the cumulative ablation bundles successful-history retention and support weighting. A fixed-support control with the same history cache and gate would directly test q_j = n_success / N_roll. Retention-coefficient sensitivity does not isolate this mechanism. Likewise, uniform and pure K-center comparators under the same final learning objective would better establish RA-Coreset's specific role.
3. The core gains need careful attribution. On OBJECT, execution supervision already reaches AUC 85.0 versus DEBR 86.5, and ALIGN alone moves RA-Coreset from 73.2 to 80.2. Explain which effects are timing correction, execution supervision, history retention and candidate selection; establish the contribution around the strongest supported mechanism. Three-seed aggregate standard deviations do not establish the significance of paired differences. Report paired seed effects when available.
4. Evaluation uses six pretraining tasks and four sequential incremental tasks, a fixed order, two simulation suites and three seeds. This supports the stated setting but leaves longer-horizon/order robustness uncertain. A targeted robustness extension is more informative than accumulating similar plots. Real-robot validation would add deployment evidence, but absence is not established here as a mandatory rejection criterion.
5. Submission preparation is incomplete: Chinese body, no abstract/reference list, SHBR/SBHR inconsistency and references to nonexistent Section 3.4.4 (gate is Section 3.4.3). OMML equations exist; plain-text extraction omissions must not be reported as missing equations. This review does not certify page layout.

## Recommended submission decision

The work has a credible RAS submission case, but current evidence does not justify calling first-round survival likely at 70–80%. Prioritize comparison provenance, contribution isolation and submission completeness. Do not launch additional experiments solely to satisfy this review or publication workflow.
