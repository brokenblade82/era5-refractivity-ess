# ERA5 refractivity: thermodynamic budgets and domain dependence

Research software and compact evidence for **Thermodynamic Error Budgets and
Domain Dependence of Radiosonde-Based ERA5 Refractivity Corrections**.

## Scientific scope

The study examines dry/wet refractivity error budgets, station/time holdouts,
and transfer to the specific RAPSODI and COSMIC-2 samples studied. The joint
holdout does not demonstrate a universal correction advantage. Improvement
in the RAPSODI sample and deterioration in the COSMIC-2 sample are retained;
they are not universal claims about either observing platform. 

## Quick verification

Use Python 3.10 or later in a separate environment:

```sh
python -m pip install -r requirements-summary.txt
python verify_summary_evidence.py
```

This checks included file hashes, algebraic error-budget closure, pooled
external error summaries reconstructed from cluster sufficient statistics,
regional station accounting, and suppression of under-supported humidity
intervals. It does not rerun training or independently reproduce every table.

## Contents and reproduction limits

* `src/` and `scripts/`: research source, including historical comparison
  methods needed to interpret the existing analysis. No Hugging Face models
  or experiments are included. Acquisition/training scripts are supplied as
  source; they require upstream data and additional scientific dependencies.
* `configs/` and `data/revision2/manifests/`: frozen protocol settings and
  station membership, not raw atmospheric observations.
* `paper_outputs/ess_reviewer_revision/`: final compact diagnostic tables.
* `results_mapping/`: selected saved metric tables, supplied for inspection.
* `COMPANION_ARCHIVE_INVENTORY.csv`: hashes and relative paths of binary
  predictions/models in the local evidence snapshot, **not files provided
  by this repository**. No public companion archive is claimed to exist yet.

Full frozen-output reproduction requires the keyed predictions, masks,
covariance outputs and models in a separately accessible companion archive.
Raw-data reproduction additionally requires upstream downloads, access terms
and dependencies. This compact repository alone is not the complete data
availability deliverable for submission. A versioned archival deposit and
its persistent identifier must be added when actually available.

Source scripts retain their original relative project paths. Do not run
training or full diagnostics expecting the compact repository to contain
their inputs. See `REPRODUCIBILITY.md` for the scope of each evidence group.

## License and citation

Original software and author-owned documentation are MIT licensed. Upstream
data and third-party rights are not relicensed; see `THIRD_PARTY_NOTICES.md`.
The repository contains aggregate evidence and station/protocol metadata,
not raw ERA5, radiosonde or radio-occultation records. Cite the original
data providers when using the corresponding results. `CITATION.cff` identifies
this software; no manuscript DOI has been assigned here.
