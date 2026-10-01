# Invalid legacy CTS output-divergence results

The following files are retained only for before/after debugging and must not be
used in paper analyses:

- `cts_dout_shard0.json`
- `cts_dout_shard1.json`
- `cts_dout_shard2.json`
- `cts_dout_shard3.json`

They were produced with `max_length=320` and right truncation. For 20 of the 30
SuperNI adapters this removed the task instance and reduced the effective input
set to one shared preamble. Their `D_out` values therefore measure continuations
from invalid conditioning inputs.

The corrected results are `cts_dout2_shard0.json` through
`cts_dout2_shard7.json`, produced with `max_length=2048`, left truncation, and
the task instance preserved. Table `tab:where` in the paper uses these corrected
files together with global Frobenius adapter strengths from
`cts_strength_aggregations.json`.
