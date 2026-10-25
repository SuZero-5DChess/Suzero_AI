import subprocess
from pathlib import Path
p=Path('/tmp/hc-design-profile')
for version in ['old','global-adaptive','new']:
 d=p/version
 cmd=['/opt/homebrew/bin/g++-15','-std=c++23','-O3','-DNDEBUG','-fno-rtti','-I'+str(p)]+['-I'+str(d/'src'/sub) for sub in ['core','misc','engine','']]+[str(p/'paths.cpp'),str(d/'src/engine/rollout_policy.cpp'),'build-gcc/lib5dchess_engine.a','build-gcc/lib5dchess_core.a','-o',str(p/('paths-'+version))]
 r=subprocess.run(cmd,capture_output=True,text=True);(p/('build-'+version+'.log')).write_text(r.stdout+r.stderr)
 print(version,r.returncode,flush=True)
