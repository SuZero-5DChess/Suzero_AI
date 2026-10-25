from pathlib import Path
p=Path('/tmp/hc-design-profile/positions');p.mkdir(exist_ok=True)
for f in Path('logs/elo-runs').glob('*/match-*/protocol-failure-*.txt'):
 text=f.read_text();marker='Partial game PGN:\n'
 if marker not in text:continue
 name=f.parent.parent.name+'-'+f.parent.name+'.5dpgn'
 (p/name).write_text(text.split(marker,1)[1])
print(len(list(p.glob('*'))),'failure PGNs extracted')
