"""Verify included aggregate evidence; no training or full-data claim."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
E = ROOT / 'paper_outputs/ess_reviewer_revision'

def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)

def main():
    manifest = pd.read_csv(ROOT/'FILE_MANIFEST.csv')
    for row in manifest.itertuples():
        p = ROOT / row.path
        assert p.stat().st_size == row.bytes, row.path
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row.sha256, row.path
    for name in ['component_budgets.csv', 'humidity_sensitivity.csv']:
        d = pd.read_csv(E/name)
        close(d.dry_mse + d.wet_mse + d.cross, d.mse)
        close(d.bias_squared + d.centered_variance, d.mse)
        close(d.rmse**2, d.mse)
    d = pd.read_csv(E/'external_error_decomposition.csv')
    close(d.bias_squared + d.centered_variance, d.mse)
    s = pd.read_csv(E/'external_cluster_sums.csv')
    keys = ['dataset','group','model']
    sums = s.groupby(keys, as_index=False)[['n','error','sse']].sum()
    joined = d.merge(sums, on=keys, suffixes=('', '_cluster'), validate='one_to_one')
    assert len(joined) == len(d)
    close(joined.n, joined.n_cluster)
    close(joined.mse, joined.sse/joined.n_cluster)
    close(joined.bias, joined.error/joined.n_cluster)
    blocks = pd.read_csv(E/'block_accounting.csv').groupby('reason').stations.sum()
    assert blocks.sum() == 630
    assert blocks['selected'] == 579 and blocks['below_minimum'] == 21
    assert blocks['outside_top20_after_southern_reservation'] == 30
    stations = pd.read_csv(E/'station_block_membership.csv')
    assert len(stations) == stations.station_id.nunique() == 630
    humidity = pd.read_csv(E/'humidity_sensitivity.csv')
    small = humidity.loc[humidity.stations < 10]
    assert len(small) == 4
    assert small[['lower','upper']].isna().all().all()
    print(f'PASS: {len(manifest)} file hashes; error budgets; external cluster sums; '
          '630-station accounting; four suppressed humidity intervals.')
    print('Scope: aggregate consistency only; full frozen-output reproduction requires companion inputs.')

if __name__ == '__main__':
    main()
