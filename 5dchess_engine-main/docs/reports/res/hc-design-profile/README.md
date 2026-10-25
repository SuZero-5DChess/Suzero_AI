# HC design profile artifacts

See [the report](../../hc-design-profile.md) for interpretation and limitations.
All commands run from the repository root. Tools generate and build only in
`/tmp/hc-design-profile`; existing production binaries are not overwritten.

Inputs:

* `test/pgn/slow/flat-ucb-match-123.5dpgn` is the original reported position.
* `positions/` contains the complete partial PGNs from historical failure logs
  `20260921-212453/match-000059` and `match-000072`.
* `production-under-profile.patch` records the working production implementation
  compared against baseline commit `b6e16ec58efd6a0c865fa0afd808f170b48794d1`.
  Reproduction assumes those working changes and matching build-gcc libraries.

Data:

* `screen.csv`: 57 fixture endpoints, two designs, one screening trial each.
* `failures.csv`: 12 historical failure endpoints, three designs, one trial each.
* `core.csv`: three rotated main comparisons, then five interleaved comparisons
  of full rebuild versus full in-place removal (`trial >= 3`).
* `paths.csv`: first end-to-end samples, including three independently reset RNG
  seeds on match 123. This predates extra normalization/destruction counters.
* `refined.csv`: repeated engine/fine-tree cases with the extra phase counters.

Paths in saved CSVs point at the durable historical inputs here. Scripts use
scratch copies at `/tmp/hc-design-profile/positions/`. Timings are milliseconds.
The method uses phase timers and counters, not statistical stack sampling.
Timeouts are soft budgets checked between attempts, so a long individual
operation or cleanup can overshoot. No action is classified as exhaustion when
this budget fires.

Prepare and build the core probe:

```sh
mkdir -p /tmp/hc-design-profile
cp docs/reports/res/hc-design-profile/*.py /tmp/hc-design-profile/
cp docs/reports/res/hc-design-profile/*.cpp /tmp/hc-design-profile/
cp docs/reports/res/hc-design-profile/profile.h /tmp/hc-design-profile/
cp -R docs/reports/res/hc-design-profile/positions /tmp/hc-design-profile/
python3 /tmp/hc-design-profile/make_probe.py
/opt/homebrew/bin/g++-15 -std=c++23 -O3 -DNDEBUG -fno-rtti \
  -I/tmp/hc-design-profile -Isrc/core -Isrc/misc -Isrc/engine -Isrc \
  /tmp/hc-design-profile/probe.cpp /tmp/hc-design-profile/geometry.cpp \
  build-gcc/lib5dchess_core.a -o /tmp/hc-design-profile/probe
python3 /tmp/hc-design-profile/run_core.py
```

The generated geometry copy adds an instrumented version of backward removal;
production geometry files are untouched. `full` mode disables the stopping and
quality limits, replacing intersecting HCs in place. The core probe validates
any returned action with `state::can_apply`.

Build and run the end-to-end probes after core measurements finish:

```sh
python3 /tmp/hc-design-profile/make_paths.py
python3 /tmp/hc-design-profile/build_paths.py
python3 /tmp/hc-design-profile/run_paths.py
python3 /tmp/hc-design-profile/run_refined.py
```

The generators copy current sources to scratch, restore baseline engine/fine-tree
files where needed, change conflict scope for the intermediate variant, and add
instrumentation only to scratch files. Replay-generated fine-tree and engine
sources were checked byte-for-byte against all six source files used for the
final measurements. The fine-tree timeout exception is injected only into this
profiling build; it is not a proposed production cancellation mechanism.

Optional initial screening uses `screen.py`. `extract.py` and
`screen_failures.py` additionally require the locally available historical
failure logs. Do not run measurements concurrently with builds or other probes.
