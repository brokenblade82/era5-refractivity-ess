# Reproduction scope

| Question | Included evidence | What the quick check verifies |
| --- | --- | --- |
| Dry/wet mechanism | component_budgets.csv | dry MSE + wet MSE + cross = total MSE; bias squared + centered variance = MSE |
| External error decomposition | external_error_decomposition.csv; external_cluster_sums.csv | pooled bias and MSE from the same cluster sums |
| Regional station membership | station_block_membership.csv; block_accounting.csv | 630 = 579 + 21 + 30 and distinct station identifiers |
| Humidity sensitivity | humidity_sensitivity.csv | budget closure and absent confidence bounds for fewer than 10 stations |
| Conditional intervals | conditional_coverage.csv; calibration_scale_quintiles.json; station_coverage.csv | saved evidence supplied; row-level reconstruction requires companion inputs |
| Correlation conversion | external_covariance_* | saved evidence supplied; conversion requires six-level distributions and masks |
| Stopping iterations | actual_iterations.csv | frozen-model inventory supplied; model inspection requires binary archive |
| Probability contrasts | keyed_probability_contrasts.csv | saved contrasts supplied; full probability reconstruction requires keyed archive |

The final humidity export suppresses four intervals supported by fewer than
10 stations; point estimates are unchanged. The bundled regeneration script
applies this reporting rule. Other historical source scripts are retained
for provenance, not presented as a newly tested end-to-end reproduction.

The quick check does not test predictive validity, rerun bootstrap inference,
prove statistical significance or demonstrate complete access to all inputs.
The companion inventory lists excluded parquet/joblib files from the frozen
local package. Upstream raw data and remaining non-binary inputs may also be
required by particular entry points. Do not interpret this inventory alone
as a ready-to-run, licensed public archive.

The reference software environment used Python 3.10.18, NumPy 1.26.4,
pandas 2.3.3, SciPy 1.15.2, scikit-learn 1.7.2, pyarrow 22.0.0 and
joblib 1.5.1. Acquisition and plotting can require xarray, netCDF backends,
PyYAML, requests, tqdm and matplotlib. These full workflows have not been
re-executed from this compact release. No external observations were used
for post-hoc calibration in preparing this repository.
