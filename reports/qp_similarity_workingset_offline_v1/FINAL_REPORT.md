# Similarity-guided constraint working-set offline v1

Primary label: **CONSTRAINT_SELECTION_WORKS_SIMILARITY_UNPROVEN**

All results below use tau=0.999. Sensitivity thresholds cannot replace this decision.

## Frozen gates

```json
{
  "H1": true,
  "H2": true,
  "H3": false,
  "H4_synthetic": true,
  "real_advance": true
}
```

## Separate synthetic and real evidence

```json
{
  "high_mixed": {
    "n": 18,
    "numerical_all": true,
    "comparison_certified": true,
    "median_k_final_fraction": 0.6171875,
    "median_T_SIM_FULL": 0.7788945989739495,
    "median_T_SIM_RANDOM": 1.086890816729302,
    "median_solves_SIM_RANDOM": 1.1884920634920635,
    "H3": false
  },
  "low": {
    "n": 9,
    "numerical_all": true,
    "comparison_certified": true,
    "median_k_final_fraction": 1.0,
    "median_T_SIM_FULL": 1.054995836645658,
    "median_T_SIM_RANDOM": 1.046425084176576,
    "median_solves_SIM_RANDOM": 1.0,
    "H3": false
  },
  "confirmation": {
    "implementation_high_mixed": {
      "n": 12,
      "numerical_all": true,
      "comparison_certified": true,
      "median_k_final_fraction": 0.6171875,
      "median_T_SIM_FULL": 0.7392455458873035,
      "median_T_SIM_RANDOM": 1.0925514367089777,
      "median_solves_SIM_RANDOM": 1.2301587301587302,
      "H3": false
    },
    "implementation_low": {
      "n": 6,
      "numerical_all": true,
      "comparison_certified": true,
      "median_k_final_fraction": 1.0,
      "median_T_SIM_FULL": 1.0577474118976848,
      "median_T_SIM_RANDOM": 1.0501666172612443,
      "median_solves_SIM_RANDOM": 1.0,
      "H3": false
    },
    "confirmation_high_mixed": {
      "n": 6,
      "numerical_all": true,
      "comparison_certified": true,
      "median_k_final_fraction": 0.609375,
      "median_T_SIM_FULL": 0.8424454250801863,
      "median_T_SIM_RANDOM": 1.0384716753867407,
      "median_solves_SIM_RANDOM": 1.0845238095238097,
      "H3": false
    },
    "confirmation_low": {
      "n": 3,
      "numerical_all": true,
      "comparison_certified": true,
      "median_k_final_fraction": 1.0,
      "median_T_SIM_FULL": 1.0503031946993824,
      "median_T_SIM_RANDOM": 1.046425084176576,
      "median_solves_SIM_RANDOM": 1.0,
      "H3": false
    }
  },
  "real_stress": {
    "n": 5,
    "numerical_all": true,
    "comparison_certified": true,
    "median_k_final_fraction": 0.6875,
    "median_T_SIM_FULL": 0.8792568010186304,
    "median_T_SIM_RANDOM": 0.9000132830826628,
    "median_solves_SIM_RANDOM": 0.92,
    "H3": false,
    "geometric_mean_T_SIM_FULL": 0.8230959535758106,
    "slowest_T_SIM_FULL": 1.0361668355480442,
    "per_fixture": [
      {
        "fixture": "B185",
        "source": "real",
        "kind": "real",
        "split": "stress",
        "certified": true,
        "comparison_certified": true,
        "k0_fraction": 0.078125,
        "k_final_fraction": 0.5625,
        "T_SIM_FULL": 0.6832762943056617,
        "T_SIM_RANDOM": 0.9000132830826628,
        "solves_SIM_RANDOM": 0.92
      },
      {
        "fixture": "B258",
        "source": "real",
        "kind": "real",
        "split": "stress",
        "certified": true,
        "comparison_certified": true,
        "k0_fraction": 0.453125,
        "k_final_fraction": 0.546875,
        "T_SIM_FULL": 0.6117224812878862,
        "T_SIM_RANDOM": 0.5544460150003702,
        "solves_SIM_RANDOM": 0.6111111111111112
      },
      {
        "fixture": "E12_B58",
        "source": "real",
        "kind": "real",
        "split": "stress",
        "certified": true,
        "comparison_certified": true,
        "k0_fraction": 0.796875,
        "k_final_fraction": 0.890625,
        "T_SIM_FULL": 0.9921015183681202,
        "T_SIM_RANDOM": 0.9968652793941237,
        "solves_SIM_RANDOM": 0.9574468085106383
      },
      {
        "fixture": "F1",
        "source": "real",
        "kind": "real",
        "split": "stress",
        "certified": true,
        "comparison_certified": true,
        "k0_fraction": 1.0,
        "k_final_fraction": 1.0,
        "T_SIM_FULL": 1.0361668355480442,
        "T_SIM_RANDOM": 1.0332469757573055,
        "solves_SIM_RANDOM": 1.0
      },
      {
        "fixture": "F2",
        "source": "real",
        "kind": "real",
        "split": "stress",
        "certified": true,
        "comparison_certified": true,
        "k0_fraction": 0.53125,
        "k_final_fraction": 0.6875,
        "T_SIM_FULL": 0.8792568010186304,
        "T_SIM_RANDOM": 0.855632776604926,
        "solves_SIM_RANDOM": 0.5882352941176471
      }
    ]
  }
}
```

## Limits

- Five selected real hard QPs all have64 valid constraints; no ordinary real32/96 set was added.
-27 synthetic fixtures are generated as A=N.T@N,b=N.T@v. The third seed provides9 confirmation cases.
- All timed methods start from zero; cached/full refined references are evaluation-only.
- RANDOM_ADD seeds0/1/2 are reported separately; paired attribution uses their median, never their best.
- FULL_WS dynamically chooses its own support. Full inner certification is reused without a redundant Decimal pass.
- SIM_ONESHOT reuses the first SIM_ADD result and never contributes an accepted solution when full certification fails.
- Time includes the inherited kernel condition-number and trace instrumentation. Extra whole-matrix spectra and reference comparisons are outside measured calls.
-10 timing repetitions measure CPU timing stability, not10 independent QPs. No GPU/backpropagation speedup or closed-loop benefit is tested.
- No float32 writeback or formal training. Original solver/recovery files remain unchanged.

`training_active=false; formal_solver_modified=false; model_writeback=false; GPU_required=false`
