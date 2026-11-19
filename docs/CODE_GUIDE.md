# ECG code and data guide

This guide describes the production functions and the decisions used by
`notebooks/EDA.ipynb`. Code is shared between Python modules, the notebook,
and the tests. Dataset provenance and licensing are in [DATA_NOTICE.md](../DATA_NOTICE.md).

## Data and selection rules

- Keep the original WFDB files unchanged in `data/raw/mitdb-1.0.0/`.
- Use channel **0**, in the physical unit read from each record header.
- A candidate contains **0.5 seconds before and after** its central annotation,
  including both endpoints. At 360 Hz it has **361 samples**, with the center at
  zero-based index **180**.
- Candidate extraction may use zero padding at recording boundaries so the
  candidate can be inspected and its rejection explained.
- **Training:** accept only **complete, clean windows with central symbol N**.
- **Validation and test:** accept **all complete windows with an eligible central
  annotation**, including windows with additional annotations.
- **Clean** means there is no additional retained annotation in the window.
  Another `N` counts as additional, including an entry at the same sample as the
  center. Entries are distinguished by their original annotation indices.
- Fit preprocessing statistics only on the windows actually accepted for
  training. Reuse those parameters for validation and test.
- Keep original annotation symbols in metadata. The pipeline does not create
  a binary target mapping, model, anomaly score, or decision threshold.

Use exactly the **19 Beat annotations** in the
[PhysioNet standard annotation table](https://physionet.org/static/lightwave/doc/annotations.html):

```text
N L R B A a J S V r F e j n E / f Q ?
```

Symbols are case-sensitive and kept unchanged. This allowlist applies both to
window centers and additional annotations. All **Non-beat annotations**, including
`+`, `~`, `|`, `x`, and `!`, and unknown symbols are ignored. They do not form
windows and do not make a window dirty. In this project, "clean" refers to the
absence of other beat annotations, not to an assessment of signal quality.
The raw files and their original entry indices are preserved. Raw source-entry
counts and accepted-window counts are therefore different quantities.

## Subject split: `scripts/create_splits.py`

The manifest [record_splits.csv](../data/record_splits.csv) has exactly three
columns: `record_id`, `subject_id`, and `split`. Read these columns as strings.
`subject_id` is a project grouping key, normally the record ID; record `202`
uses subject `201`, as records `201` and `202` belong to the same person.

The fixed allocation is 29 training, 9 validation, and 9 test subjects. Record
100 belongs to training. Sort the other 46 subject IDs, permute them with
`numpy.random.default_rng(42)`, and assign 28/9/9. All records of a person
inherit that person's split. In the saved manifest, the shared 201/202 subject
is in training, producing 30/9/9 recordings.

| Function | Parameters | Result and behavior |
|---|---|---|
| `build_record_splits(record_ids, seed=42)` | All 48 source record IDs; random seed. | Validated split DataFrame. Preserves input row order; subject assignments are independent of that order. |
| `validate_record_splits(splits, record_ids)` | Manifest and original record IDs. | Returns `None`; raises `ValueError` for incorrect coverage, labels, subject mapping, leakage, sizes, or the assignment of record 100. |
| `main()` | No arguments; uses the project data paths. | Creates the seed-42 CSV or verifies an identical existing file. A differing existing file is preserved and causes an error. |

Reproduce the manifest:

```bash
uv run --locked python scripts/create_splits.py
```

## Window objects and extraction: `scripts/windows.py`

`WindowAnnotation` retains the original `annotation_index`, absolute `sample`,
`symbol`, auxiliary note, and signed `offset_samples` from the window center.
The original index is never renumbered when source entries are excluded.

`ECGWindow` holds a 1D physical signal copy, sampling rate, record and channel
identity, central annotation, additional annotations, and left/right padding
counts. Window properties are:

| Property | Meaning |
|---|---|
| `center_index` | Position of the center within the signal array, in samples. |
| `is_complete` | `True` when neither end requires artificial padding. |
| `has_other_annotations` | `True` when at least one additional retained annotation is present. |

| Function | Parameters | Result and behavior |
|---|---|---|
| `extract_ecg_window(record, annotations, annotation_index, channel=0)` | Loaded WFDB record and annotations with the same sample origin; original annotation-entry index; channel index. | One candidate `ECGWindow`. Bounds are inclusive. Missing source samples are zeros, and the padding counts identify them. Unsupported centers cause an error. |
| `generate_ecg_windows(record, annotations, annotation_symbol="N", channel=0)` | Matching record/annotation pair; exact central symbol; channel. | Lazy iterator in source annotation order. It selects the center by symbol and retains all eligible neighboring entries. It does not apply final dataset-quality filters. |

## Selection policy: `scripts/rules.py`

`BEAT_ANNOTATION_SYMBOLS` is the immutable allowlist above.
`is_beat_annotation(symbol)` accepts an original WFDB symbol and returns whether
it belongs to this list. Extraction, plotting, dataset preparation, and notebook
summaries all use this predicate.

`window_rejection_reasons(window, split)` accepts an extracted candidate and a
split name, then returns every failed quality or training-reference rule.
An empty tuple means the candidate is accepted. The function is shared by the
training-window reader, statistics fitting, and final dataset preparation.

Reasons include `train_center_not_N`, `incomplete_window`, and
`train_other_annotations`. The neighbor rule applies only to training.
Source symbols outside the allowlist are rejected before extraction with
`excluded_annotation_symbol` in the dataset audit.
When multiple rules fail, the audit joins their names with `;` and counts
the combined case once. This prevents double-counting rejected entries.

## Preprocessing: `scripts/preprocessing.py`

Statistics pool all sample occurrences in accepted training windows. An overlap
between windows therefore contributes once per window. Real zero amplitudes are
included. Standard deviation uses **`ddof=0`**. Statistics are global to the
reference population rather than separately fitted to each time position or
each window.

`TrainingStatistics` stores mean, median, std, sample/window counts, contributing
record IDs, channel, and original unit. Records without accepted N windows
contribute no samples. The whitelist of training record IDs is taken from the
validated manifest.

| Function | Parameters | Result and behavior |
|---|---|---|
| `fit_training_statistics(windows, training_record_ids)` | Iterable of accepted clean, complete N windows; allowed training IDs. | `TrainingStatistics`. Rejects held-out records, other central symbols, incomplete/dirty windows, mixed channels/units, empty inputs, and invalid signals. |
| `preprocess_window(window, statistics=None, mode="none")` | Candidate window; fitted statistics for centering/scaling; mode. | Tuple `(new_window, state)`. `none` copies unchanged, `center` subtracts the training mean, and `standardize` also divides by training std. Original data and annotation metadata are preserved. |
| `inverse_preprocess_window(window, state)` | Processed window and the state returned by its forward call. | A new window restored to physical values and its original unit, up to floating point rounding. |
| `_signal_and_real_slice(window)` | Window signal and padding metadata. | Validated 1D signal and slice of real samples; rejects malformed padding or nonfinite values. Used internally by the transforms. |

`PreprocessingState` stores mode, offset, scale, original unit, and channel.
Save the state with the transformed row to enable inversion. Standardized
values are dimensionless. Padding stays zero when transforming padded
candidates for inspection; final datasets contain complete windows only.
A zero std causes a standardization error rather than an implicit fallback.

## Dataset preparation: `scripts/datasets.py`

| Function | Parameters | Result and behavior |
|---|---|---|
| `iter_training_n_windows(raw_directory, training_record_ids)` | Raw WFDB directory and training record IDs from the manifest. | Lazy iterator of channel-0 N windows passing all training selection rules. |
| `prepare_window_datasets(raw_directory, record_splits, source_record_ids, statistics, mode="standardize")` | Raw directory, validated manifest, original IDs, fitted clean-training statistics, and transform mode. | Tuple `(datasets, audit, reasons)`. Uses the shared extractor and selection rules; transforms accepted windows without refitting parameters. |

`datasets` has keys `train`, `validation`, and `test`. Each `PreparedSplit` has:

- `X`: float matrix shaped `(accepted_window_count, 361)`.
- `metadata`: one source description per matrix row, in the same order.
- `states`: one inverse-transform state per row.

The arrays and metadata are held in memory. Run the notebook preparation cell
to recreate them. An empty split retains shape `(0, 361)`.

The per-symbol audit records source entries, accepted windows, rejected entries,
and the shape of each accepted symbol subset. Source entries include excluded
markers for accounting; these entries never produce windows. `reasons` reports
counts by split, original symbol, and exact rejection-reason combination.

For every split and symbol:

```text
Source entries = Accepted windows + Rejected entries
Rejected entries = sum of counts for exact rejection-reason combinations
```

Row metadata contains record/subject IDs, channel index/name, original
annotation index/sample/symbol, sampling rate, source interval, completeness,
neighbor flag, padding counts, and units. `window_stop_sample` is exclusive.
For row `i`, read `metadata.iloc[i]`, re-read the source record and annotations,
extract the central entry by its original index, and use `states[i]` to invert
the processed signal. This preserves the link to the original `.atr` file.

## Plotting: `scripts/plotting.py`

| Function | Parameters | Result and behavior |
|---|---|---|
| `plot_ecg_with_annotations(record, annotations, channel=0, start_s=0, duration_s=10, layout="overlay")` | Matching source record/annotation pair; one channel or a pair of channel indices; start/duration in seconds; layout. | Matplotlib figure and axes. `overlay` shares one axis; `stacked` uses one row per channel. The interval is half-open. Symbols are placed to the right of markers. |
| `plot_ecg_windows_grid(windows, ncols=4)` | Nonempty sequence of windows and maximum column count. | Static figure and a 2D axes array; plots every supplied window. Select a small list such as `windows[:20]`. Shared amplitude limits use the supplied windows, and unused panels are hidden. |

Both plotting functions return objects for display with `plt.show()` or export
with `fig.savefig(...)`. The grid uses relative time, marks the center at 0 s,
and preserves the displayed signal unit.

## Setup and data restoration

| Function | Parameters | Result and behavior |
|---|---|---|
| `fetch_data.checksum(data)` | Raw bytes. | Hexadecimal SHA-256 digest. |
| `fetch_data.main()` | No arguments; fixed project paths. | Restores/verifies original raw files and the manifest, with retrieval progress and failure details. |
| `fetch_data.main.fetch(name)` | Raw filename, including extension. | Internal worker returning source URL, size, and checksum; preserves mismatching local files. |
| `verify_setup.main()` | No arguments; pinned environment and project data. | Checks dependency imports, Python version, 144 raw file sizes/checksums, record shape/rate, and annotation bounds. |

```bash
uv sync --locked
uv run --locked python scripts/verify_setup.py
```

Restore missing data only when needed:

```bash
uv run --locked python scripts/fetch_data.py
```

## Notebook navigation

Cell numbers count every cell from the top, including Markdown. Line numbers
refer to lines inside a code cell, not the notebook JSON file.
After changing project modules, restart the notebook kernel and run all cells
to load the updated definitions and recreate the in-memory datasets.

| Cell | Purpose |
|---|---|
| 2 | Project-root detection and loading record 100. |
| 3 | Annotation-symbol counts for the example. |
| 5, 7, 9 | Signal plot import and overlay/stacked examples. |
| 11–14 | Window API imports and example candidate windows. |
| 16–17 | Static grid import and first-20 example. |
| 19 | Candidate N-window counts across records. |
| 21 | Saved split validation and subject-isolation check. |
| 23 | Candidate-window comparison by split and central symbol. |
| 25–26 | Training-window reader, preprocessing API imports, and fitted statistics. |
| 28 | Twenty random training-reference windows from record 100, seed 42. |
| 30 | Final matrices and selection audits; mode is selected on line 3. |
| 32 | Trace one matrix row back to its source; split/row are on lines 3–4. |

## Verification

```bash
uv run --locked python -m unittest discover -s tests -v
```

Tests cover split reproducibility and subject leakage, preprocessing/inversion,
training-only fitting, final selection, rejection accounting, empty matrix
shapes, inclusive window boundaries, all approved beat symbols, all excluded
non-beat symbols, plot labels, and reconstruction of source rows. Test fixture helpers are documented
in their test files. Notebook cells also expose source metadata and inverse
errors for inspection. No model-evaluation results are implied by these checks.
