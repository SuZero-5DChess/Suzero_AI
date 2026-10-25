from pathlib import Path
import subprocess,csv
p=Path('/tmp/hc-design-profile')
header='path,partition,mode,result,elapsed_ms,build_ms,point_ms,problem_ms,remove_ms,attempts,empty,problems,hc_visits,cuts,quality_skips,peak_hcs,remaining_hcs'
paths=list(Path('test/pgn').glob('*.5dpgn'))+list(sorted(Path('test/pgn/online').glob('*.5dpgn')))[::20]+list(Path('test/pgn/slow').glob('*.5dpgn'))
with (p/'screen.csv').open('w') as f:
 f.write(header+'\n');f.flush()
 for path in paths:
  for mode in ['local-adaptive','global-full-rebuild']:
   r=subprocess.run([str(p/'probe'),str(path),mode,'300'],capture_output=True,text=True,timeout=15)
   if r.returncode: print(path,mode,r.returncode,r.stderr,flush=True);continue
   f.write(r.stdout);f.flush()
rows=list(csv.DictReader((p/'screen.csv').open()))
for r in sorted(rows,key=lambda r:float(r['elapsed_ms']),reverse=True)[:20]:print(r['path'],r['mode'],r['result'],r['elapsed_ms'],flush=True)
