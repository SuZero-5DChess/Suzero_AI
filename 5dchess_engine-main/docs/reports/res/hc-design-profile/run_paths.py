from pathlib import Path
import subprocess
p=Path('/tmp/hc-design-profile')
cases=['test/pgn/slow/flat-ucb-match-123.5dpgn',str(p/'positions/20260921-212453-match-000072.5dpgn'),str(p/'positions/20260921-212453-match-000059.5dpgn'),'test/pgn/niceAndrey.5dpgn']
with (p/'paths.csv').open('w') as f:
 f.write('version,path,mode,seed,result,elapsed_ms,point_ms,problem_ms,remove_ms,attempts,empty,problems,setup_ms,isolate_ms,normalize_ms,destroy_ms\n')
 for path in cases:
  seeds=[7,42,123] if '123.5dpgn' in path else [42]
  for seed in seeds:
   for mode in ['random','weighted','fine-natural','fine-random']:
    for version in ['old','global-adaptive','new']:
     r=subprocess.run([str(p/('paths-'+version)),path,mode,'2000',str(seed)],capture_output=True,text=True,timeout=15)
     if r.returncode:raise RuntimeError(str((version,path,mode,r.returncode,r.stderr)))
     f.write(version+','+r.stdout);f.flush()
     row=r.stdout.strip().split(',');print(Path(path).name,seed,mode,version,row[3],row[4],flush=True)
