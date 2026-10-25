from pathlib import Path
import subprocess,csv
p=Path('/tmp/hc-design-profile-allgood');old=Path('/tmp/hc-design-profile');slow='test/pgn/slow/flat-ucb-match-123.5dpgn'
cases=[(slow,'all'),(slow,'restricted'),(str(old/'positions/20260921-212453-match-000072.5dpgn'),'all'),(str(old/'positions/20260921-212453-match-000059.5dpgn'),'all'),('test/pgn/niceAndrey.5dpgn','all'),('test/pgn/100timelines.5dpgn','all')]
modes=['local-adaptive','global-adaptive','global-codim1-all','global-full-rebuild']
header=(old/'core.csv').read_text().splitlines()[0]
with (p/'core.csv').open('w') as f:
 f.write(header+'\n')
 for case,partition in cases:
  for trial in range(3):
   order=modes[trial:]+modes[:trial]
   for mode in order:
    r=subprocess.run([str(p/'probe'),case,mode,'5000',partition],capture_output=True,text=True,timeout=20)
    if r.returncode:raise RuntimeError(str((case,mode,r.stderr)))
    f.write(str(trial)+','+r.stdout);f.flush()
    row=r.stdout.strip().split(',');print(Path(case).name,partition,trial,mode,row[3],row[4],flush=True)
versions=['old','global-adaptive','global-codim1-all','new']
pathcases=[slow,cases[2][0],cases[3][0],cases[4][0],cases[5][0]]
with (p/'paths.csv').open('w') as f:
 f.write('trial,version,path,mode,seed,result,elapsed_ms,point_ms,problem_ms,remove_ms,attempts,empty,problems,setup_ms,isolate_ms,normalize_ms,destroy_ms\n')
 for case in pathcases:
  for mode in ['fine-natural','fine-random','random','weighted']:
   for trial in range(3):
    order=versions[trial:]+versions[:trial]
    for version in order:
     r=subprocess.run([str(p/('paths-'+version)),case,mode,'2000','42'],capture_output=True,text=True,timeout=20)
     if r.returncode:raise RuntimeError(str((case,mode,version,r.stderr)))
     f.write(str(trial)+','+version+','+r.stdout);f.flush()
   print('paths',Path(case).name,mode,'done',flush=True)
