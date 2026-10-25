from pathlib import Path
import subprocess
p=Path('/tmp/hc-design-profile')
header='trial,version,path,mode,seed,result,elapsed_ms,point_ms,problem_ms,remove_ms,attempts,empty,problems,setup_ms,isolate_ms,normalize_ms,destroy_ms'
cases=[(str(p/'positions/20260921-212453-match-000072.5dpgn'),'fine-natural',42),(str(p/'positions/20260921-212453-match-000072.5dpgn'),'fine-random',42),(str(p/'positions/20260921-212453-match-000059.5dpgn'),'weighted',42),(str(p/'positions/20260921-212453-match-000059.5dpgn'),'fine-natural',42)]
with (p/'refined.csv').open('w') as f:
 f.write(header+'\n')
 for path,mode,seed in cases:
  for trial in range(3):
   versions=['old','global-adaptive','new'];versions=versions[trial:]+versions[:trial]
   for version in versions:
    r=subprocess.run([str(p/('paths-'+version)),path,mode,'5000',str(seed)],capture_output=True,text=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr)
    f.write(str(trial)+','+version+','+r.stdout);f.flush()
  print(Path(path).name,mode,'done',flush=True)
 # Check the weighted seed that exceeds a 200 ms move budget independently.
 for trial in range(3):
  r=subprocess.run([str(p/'paths-new'),'test/pgn/slow/flat-ucb-match-123.5dpgn','weighted','5000','42'],capture_output=True,text=True,timeout=20)
  if r.returncode:raise RuntimeError(r.stderr)
  f.write(str(trial)+',new,'+r.stdout);f.flush()
