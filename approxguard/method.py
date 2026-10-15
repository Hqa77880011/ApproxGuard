import numpy as np
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.get('input_shape',[8])[-1];self.net=nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,b):return {'prediction':self.net(b['x'].float())}
    def objective(self,b,p):return F.mse_loss(p['prediction'],b['y'].float())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(0,1,(n,8)).astype('float32');return {'x':x,'y':(x.sum(1,keepdims=True)/8).astype('float32'),'group':np.arange(n)}


def mapping(problem,variant='full'):
    import numpy as np
    multipliers=problem['multipliers'];layers=problem['layers'];clock=problem['clock'];ages=np.asarray(problem['ages']);pvt=np.asarray(problem['pvt'])
    timing=[];risk=[]
    for m in multipliers:
        delay=m['delay']*(1+m['aging']*ages[:,None]+pvt[None]);timing.append(np.mean(delay>clock));risk.append(np.maximum(delay-clock,0).max())
    selected=[]
    for layer in layers:
        sensitivity=1 if variant=='no_sensitivity' else layer['sensitivity'];scores=[]
        for j,m in enumerate(multipliers):
            penalty=sensitivity*(m['med']+m['error_rate']+layer['exposure']*m['error_rate']);aging=0 if variant in {'baseline','no_aging'} else risk[j]*layer['exposure'];scores.append(m['energy']+problem.get('area_weight',.1)*m['area']+penalty+aging)
        selected.append(int(np.argmin(scores)))
    def evaluate(indices):
        key=','.join(map(str,indices));estimated=problem['baseline_accuracy']-sum(layers[l]['sensitivity']*multipliers[j]['error_rate'] for l,j in enumerate(indices));accuracy=problem.get('validation_accuracy',{}).get(key,estimated);failure=float(np.mean([timing[j] for j in indices]));return accuracy,failure
    while True:
        accuracy,failure=evaluate(selected)
        if accuracy>=problem['baseline_accuracy']-problem['accuracy_tolerance'] and failure<=problem['timing_tolerance']:break
        candidates=[l for l,j in enumerate(selected) if j!=0]
        if not candidates:break
        layer=max(candidates,key=lambda l:layers[l]['sensitivity']*multipliers[selected[l]]['error_rate']+risk[selected[l]]);selected[layer]=0
    accuracy,failure=evaluate(selected)
    return {'mapping':selected,'accuracy':accuracy,'accuracy_source':'whole_configuration_validation' if ','.join(map(str,selected)) in problem.get('validation_accuracy',{}) else 'additive_profile_model','timing_violation_fraction':failure,'energy_proxy':sum(multipliers[j]['energy'] for j in selected),'area_proxy':sum(multipliers[j]['area'] for j in selected),'feasible':accuracy>=problem['baseline_accuracy']-problem['accuracy_tolerance'] and failure<=problem['timing_tolerance']}


def execute(args):
    import json
    from pathlib import Path
    result=mapping(json.loads(Path(args.input).read_text()),args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'mapping.json').write_text(json.dumps(result,indent=2))
