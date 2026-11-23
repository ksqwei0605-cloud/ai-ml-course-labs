import json,subprocess,sys
from pathlib import Path
q=Path(__file__).resolve().parents[1]/'code/q2_clustering'
root=q/'outputs/sensitivity_v1'
cfg=json.loads((root/'raw.json').read_text())
cfg['silhouette_sample_size']=56000
p=root/'full_eval.json'
p.write_text(json.dumps(cfg,indent=2))
dest=root/'full_eval'
subprocess.run([sys.executable,'extras.py','evaluate','--device','cuda','--experiment',str(root/'raw'),'--config',str(p),'--output',str(dest)],cwd=q,check=True)
with (root/'full_eval_check.json').open('w') as f: subprocess.run([sys.executable,'extras.py','check-output','--output',str(dest)],cwd=q,stdout=f,check=True)
print((dest/'selection.json').read_text())
