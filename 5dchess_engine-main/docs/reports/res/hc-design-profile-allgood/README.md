# All-HC, codimension ≤1 profile extension

See [the extended report](../../hc-design-profile.md).

* `core.csv`: 72 trials (six spaces × four policies × three repetitions).
* `paths.csv`: 240 trials (five positions × four operations × four policies ×
  three repetitions). Fine-tree elapsed time includes destruction; the report
  subtracts each trial's `destroy_ms` before taking the median.
* `extend_allgood.py`: creates scratch sources and builds all four policies
  against the same scratch geometry. It does not change production code.
* `run_extended.py`: runs the trials sequentially, resetting RNG seed 42.
* `summarize.py`: regenerates report values from the saved CSVs.
* `validate.cpp`: small complete-action-set comparison derived from
  `test/test_global_search.cpp`, using the scratch all-HC engine/fine-tree and
  geometry sources. It passed with assertions enabled (150 and 146 actions;
  natural ordering and random seeds 7 and 42).
* `weights.cpp` / `weights.csv`: separate breakdown of `100timelines` coordinate
  scoring and weighted-order preparation; no search is performed by this probe.

To reproduce, prepare `/tmp/hc-design-profile` using the source-copy commands in
[the original artifact README](../hc-design-profile/README.md), then run its
`make_probe.py` and `make_paths.py`. Compiling or running the original probes is
not required. The original production patch and baseline commit requirements
still apply.

From the repository root:

```sh
python3 docs/reports/res/hc-design-profile-allgood/extend_allgood.py
python3 docs/reports/res/hc-design-profile-allgood/run_extended.py
```

The extension writes only to `/tmp/hc-design-profile-allgood`. To regenerate the
summary using fresh data, copy its `core.csv` and `paths.csv` into this artifact
directory and run `summarize.py`. Saved paths to historical inputs were changed
from scratch locations to their durable copies in the original artifact folder.

The all-HC variant always cuts the selected HC; secondary cuts use codimension
≤1. In fine-tree, all cells on the ancestor/context path are visited, without
quality-based ancestor early termination. Engine all-HC propagation starts at
the first conflict, while original/global-with-stop engines retain the original
100-problem iterative phase. Consult the report before comparing those engine
columns as if only one setting had changed.

The fine-tree budget exception exists only in scratch profiling builds. This
is not a production stop-token or cancellation change.

Build the separate coordinate-weight preparation probe after all other
measurements have finished:

```sh
/opt/homebrew/bin/g++-15 -std=c++23 -O3 -DNDEBUG -fno-rtti \
  -Isrc/core -Isrc/misc -Isrc/engine -Isrc \
  docs/reports/res/hc-design-profile-allgood/weights.cpp \
  build-gcc/lib5dchess_engine.a build-gcc/lib5dchess_core.a \
  -o /tmp/hc-design-profile-allgood/weights
/tmp/hc-design-profile-allgood/weights
```

To rerun the action-set comparison after building the scratch sources:

```sh
/opt/homebrew/bin/g++-15 -std=c++23 -O3 -DNDEBUG -fno-rtti \
  -I/tmp/hc-design-profile-allgood \
  -I/tmp/hc-design-profile-allgood/global-codim1-all/src/core \
  -I/tmp/hc-design-profile-allgood/global-codim1-all/src/misc \
  -I/tmp/hc-design-profile-allgood/global-codim1-all/src/engine \
  -I/tmp/hc-design-profile-allgood/global-codim1-all/src \
  docs/reports/res/hc-design-profile-allgood/validate.cpp \
  /tmp/hc-design-profile-allgood/global-codim1-all/src/engine/rollout_policy.cpp \
  /tmp/hc-design-profile-allgood/geometry.cpp \
  build-gcc/lib5dchess_engine.a build-gcc/lib5dchess_core.a \
  -o /tmp/hc-design-profile-allgood/validate
/tmp/hc-design-profile-allgood/validate
```
