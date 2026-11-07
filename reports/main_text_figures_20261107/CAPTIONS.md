# Main-text figure captions

## Figure 1. Continual closed-loop performance

Closed-loop success rates on LIBERO-OBJECT and LIBERO-GOAL after each incremental training stage. Left: the average over all seen incremental tasks. Right: the average over previously learned incremental tasks. Lines and shaded regions show the mean and one sample standard deviation across three seeds, respectively. Execution Supervision (Matched Budget) extends the execution-supervision baseline to match the physical optimizer-call budget of Ours. Support-Weighted Retention adds the retention mechanism without candidate gating; Ours includes gating. Each task-stage cell uses 20 formal initial configurations per seed. Old-task performance is undefined at the first incremental stage. The plotted tasks are 6–9; the task set entering each average grows with the training stage.

## Figure 2. Candidate tradeoffs and gating decisions

Training-time candidate diagnostics for Ours at the default retention coefficient, using all 27,300 OBJECT and 20,100 GOAL decision blocks across three seeds. Left: execution-error and retention-error changes of the support-weighted retention candidate relative to the execution-supervision candidate. Both candidates start from the same training state and use the same batch sequence. Changes are expressed as the symmetric relative difference, 200 times the candidate-error difference divided by the sum of their absolute errors; the change is zero when both errors are zero. Negative values indicate lower error for the retention candidate. Both axes use a symmetric-log scale with a 0.01% linear threshold. Right: mutually exclusive combinations of violated gate conditions, as percentages of all decision blocks. Acceptance requires non-increasing behavioral-cloning (BC) and execution errors and strictly decreasing retention error. A candidate in the lower-left scatter region may still be rejected if its BC error increases. Colors indicate acceptance or rejection of the retention candidate; they do not denote closed-loop success or failure.

## Figure 3. Task-level gains throughout continual learning

Task-stage success-rate differences between Ours and Execution Supervision (Matched Budget), and between Ours and Support-Weighted Retention (No Gate), on LIBERO-OBJECT and LIBERO-GOAL. Each cell is averaged across three seeds using paired formal initial configurations. Positive values favor Ours; negative values favor the comparator. All observed positive, zero and negative differences are retained, with a common zero-centered color scale from −25 to +25 percentage points. Gray cells denote tasks not yet introduced and are not zero-valued observations.

## Insertion notes

The descriptive names denote the same configurations as the ablation tables. In Figure 1 the leading plus sign follows the component-table convention; in Figure 3 it is omitted after the subtraction sign for readability. The support-weighted retention comparator has no gate. Local candidate branches in Figure 2 are not independently trained methods. Candidate-gating diagnostics and end-to-end success comparisons are different evidence types.

Use the PDF files at approximately 180–183 mm width in a double-column manuscript, and attach the corresponding caption below each graphic. The graphics contain panel labels but no fixed manuscript figure numbers. The captions may be shortened to fit a venue while preserving metric definitions and the distinction between local candidates and complete methods. The original full six-panel performance and gate-activity figures remain available for supplementary material; no original file was overwritten.
