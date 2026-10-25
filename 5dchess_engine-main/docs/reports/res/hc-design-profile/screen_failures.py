from pathlib import Path
import subprocess,csv
p=Path('/tmp/hc-design-profile')
with (p/'failures.csv').open('w') as f:
 f.write((p/'screen.csv').read_text().splitlines()[0]+'\n')
 for path in sorted((p/'positions').glob('*')):
  for mode in ['local-adaptive','global-adaptive','global-full-rebuild']:
   r=subprocess.run([str(p/'probe'),str(path),mode,'500'],capture_output=True,text=True,timeout=15)
   if r.returncode:print(path.name,mode,r.returncode,r.stderr[:300],flush=True);continue
   f.write(r.stdout);f.flush()
   row=r.stdout.strip().split(',');print(path.name,mode,row[3],row[4],flush=True)
