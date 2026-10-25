from pathlib import Path
import shutil,subprocess
old=Path('/tmp/hc-design-profile');p=Path('/tmp/hc-design-profile-allgood');p.mkdir(exist_ok=True)
for v in ['old','global-adaptive','new']:
 shutil.copytree(old/v,p/v,dirs_exist_ok=True)
shutil.copytree(old/'global-adaptive',p/'global-codim1-all',dirs_exist_ok=True)
for name in ['profile.h','paths.cpp','probe.cpp','geometry.cpp','geometry.h']:
 shutil.copy2(old/name,p/name)
f=p/'geometry.h';s=f.read_text().replace('size_t &skips);','size_t &skips, bool scan_all=false, bool force_back_removal=true);');f.write_text(s)
f=p/'geometry.cpp';s=f.read_text().replace('size_t &skips) {','size_t &skips, bool scan_all, bool force_back_removal) {').replace('bool force=true;','bool force=force_back_removal;').replace('(!full && disjoint*10>=intersections)','(!full && !scan_all && disjoint*10>=intersections)');f.write_text(s)
f=p/'probe.cpp';s=f.read_text().replace('space.profile_remove(*problem,full,visits,cuts,skips);','space.profile_remove(*problem,full,visits,cuts,skips,mode=="global-codim1-all");');f.write_text(s)
f=p/'profile.h';s=f.read_text().replace('attempts=0,empty=0,problems=0;','attempts=0,empty=0,problems=0,hc_visits=0,cuts=0,skips=0;');f.write_text(s)
# Fine-tree: all residual HCs on the existing ancestor/context path, quality <= 1.
f=p/'global-codim1-all/src/core/finetree.inl';s=f.read_text().replace('cell->subspace.remove_slice_backwards(s, force_back_removal);','cell->subspace.profile_remove(s, false, profile::hc_visits, profile::cuts, profile::skips, true, force_back_removal);')
s=s.replace('    if(!origin_cell->space.is_slice_good(s))\n    {\n        return;\n    }\n','').replace('        if(!path_cell->space.is_slice_good(s))\n        {\n            break;\n        }\n','')
f.write_text(s)
# Engine: match replacement's global search from the first conflict; quality-limit all secondary cuts.
s=(p/'new/src/engine/rollout_policy.cpp').read_text()
a=s.index('        // Match the core search');b=s.index('        space = std::move(remaining);',a);b=s.index('\n',b)
s=s[:a]+'''        ++profile::problems;
        space.push_back(std::move(hc));
        profile::timed(profile::remove_ms, [&]{
            space.profile_remove(*problem, false, profile::hc_visits, profile::cuts, profile::skips, true);
        });'''+s[b:]
(p/'global-codim1-all/src/engine/rollout_policy.cpp').write_text(s)
# Compile every variant against the same geometry implementation.
base=['/opt/homebrew/bin/g++-15','-std=c++23','-O3','-DNDEBUG','-fno-rtti','-I'+str(p)]
r=subprocess.run(base+['-Isrc/core','-Isrc/misc','-Isrc/engine','-Isrc',str(p/'probe.cpp'),str(p/'geometry.cpp'),'build-gcc/lib5dchess_core.a','-o',str(p/'probe')],capture_output=True,text=True)
(p/'build-core.log').write_text(r.stdout+r.stderr);print('core',r.returncode,flush=True)
if r.returncode:raise SystemExit(1)
for v in ['old','global-adaptive','global-codim1-all','new']:
 d=p/v
 cmd=base+['-I'+str(d/'src'/sub) for sub in ['core','misc','engine','']]+[str(p/'paths.cpp'),str(d/'src/engine/rollout_policy.cpp'),str(p/'geometry.cpp'),'build-gcc/lib5dchess_engine.a','build-gcc/lib5dchess_core.a','-o',str(p/('paths-'+v))]
 r=subprocess.run(cmd,capture_output=True,text=True);(p/('build-'+v+'.log')).write_text(r.stdout+r.stderr)
 print(v,r.returncode,flush=True)
 if r.returncode:raise SystemExit(1)
