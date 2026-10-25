from pathlib import Path
import subprocess,random
p=Path('/tmp/hc-design-profile');slow='test/pgn/slow/flat-ucb-match-123.5dpgn'
cases=[(slow,'all'),(slow,'restricted'),(str(p/'positions/20260921-212453-match-000072.5dpgn'),'all'),(str(p/'positions/20260921-212453-match-000059.5dpgn'),'all'),('test/pgn/niceAndrey.5dpgn','all'),('test/pgn/100timelines.5dpgn','all')]
modes=['local-adaptive','global-adaptive','global-full-rebuild']
header=(p/'screen.csv').read_text().splitlines()[0]
with (p/'core.csv').open('w') as f:
 f.write('trial,'+header+'\n');f.flush()
 for case,partition in cases:
  for trial in range(3):
   order=modes[trial:]+modes[:trial]
   for mode in order:
    r=subprocess.run([str(p/'probe'),case,mode,'5000',partition],text=True,capture_output=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr)
    f.write(str(trial)+','+r.stdout);f.flush()
    row=r.stdout.strip().split(',');print(Path(case).name,partition,trial,mode,row[3],row[4],flush=True)
 # Same semantics, adapted in-place helper versus rebuild; five interleaved trials.
 for case,partition in cases:
  for trial in range(5):
   order=['global-full-inplace','global-full-rebuild'];
   if trial%2:order.reverse()
   for mode in order:
    r=subprocess.run([str(p/'probe'),case,mode,'10000',partition],text=True,capture_output=True,timeout=25)
    if r.returncode:raise RuntimeError(r.stderr)
    f.write(str(trial+3)+','+r.stdout);f.flush()
  print('in-place comparison done:',Path(case).name,partition,flush=True)
