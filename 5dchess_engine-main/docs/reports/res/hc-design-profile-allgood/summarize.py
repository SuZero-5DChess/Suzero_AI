from pathlib import Path
import csv,collections,statistics
p=Path(__file__).resolve().parent
labels={'local-adaptive':'Old','global-adaptive':'Global + stop','global-codim1-all':'Global ≤1, all HCs','global-full-rebuild':'Full replacement','old':'Old','new':'Full replacement'}
def grouped(name,mode_key):
 g=collections.defaultdict(dict)
 for r in csv.DictReader((p/name).open()):
  key=(Path(r['path']).name,r.get('partition',r.get('mode')))
  g[key].setdefault(r[mode_key],[]).append(r)
 for variants in g.values():
  for rows in variants.values():assert len(rows)==3
 return g
def val(rows,field='elapsed_ms',subtract=None):
 if all(r['result']=='timeout' for r in rows):return '>5000' if 'partition' in rows[0] else '>2000'
 assert all(r['result']!='timeout' for r in rows)
 return f"{statistics.median(float(r[field])-(float(r[subtract]) if subtract else 0) for r in rows):.3f}"
g=grouped('core.csv','mode')
print('CORE: search excluding HC build, ms')
for (path,partition),variants in g.items():print(path,partition,*(val(variants[v]) for v in ['local-adaptive','global-adaptive','global-codim1-all','global-full-rebuild']),sep=' | ')
g=grouped('paths.csv','version')
for mode in ['fine-natural','fine-random','random','weighted']:
 print('\n'+mode+': including setup, excluding fine-tree destruction, ms')
 for (path,kind),variants in g.items():
  if kind==mode:print(path,*(val(variants[v],subtract='destroy_ms') for v in ['old','global-adaptive','global-codim1-all','new']),sep=' | ')
