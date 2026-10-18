import ast, collections, time, numpy as np, torch, contextlib, sys
src=open(sys.argv[1]).read(); tree=ast.parse(src)
g={'np':np,'collections':collections,'time':time,'torch':torch}
for node in tree.body:
    if (isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('SortedWindowQuantile','b2_thresholds','b2_decide','e2e_pass')) or (isinstance(node,ast.Assign) and getattr(node.targets[0],'id','')=='E2E_VARIANTS'):
        exec(compile(ast.Module([node],[]),'x','exec'),g)
for N in (150, 1000):
    rng=np.random.default_rng(N); scores=rng.standard_normal(N).astype(np.float32)
    class Img:
        def __init__(s,i): s.i=i
        def float(s): return torch.tensor([s.i*255.0])
    H={'img':torch.zeros(1,dtype=torch.float64),'sa':torch.zeros(1),'thr':{'a3g':torch.zeros(1)},'a3g':(torch.zeros(1),torch.zeros(1),torch.zeros(1))}
    calls=collections.Counter()
    def a3(mode):
        def f():
            calls[mode]+=1
            i=int(round(H['img'][0].item())); s=scores[i]; thr=np.float32(H['thr']['a3g'][0].item())
            use=(s>thr) if mode<0 else bool(mode)
            H['a3g'][1].fill_(1.0 if use else 0.0); H['a3g'][2].fill_(float(s))
        return f
    ops={'vanilla':lambda:None,'p2static':lambda:None,('a3g',-1.0):a3(-1),('a3g',1.0):a3(1),('a3g',0.0):a3(0)}
    batches=[(Img(i),0.0) for i in range(N)]
    lat,use=g['e2e_pass'](ops,H,batches,0.5,200,20,np.float32(-0.3),30,3,contextlib.nullcontext,lambda:None)
    ref,_=g['b2_decide'](scores,0.5,200,20,np.float32(-0.3))
    for k,v in use.items(): print(N, k, 'beda vs b2_decide =', int((np.array(v)!=ref).sum()), 'aktivasi %.3f'%np.mean(v), 'ref %.3f'%ref.mean())
    print(' n per varian', {k:len(v) for k,v in lat.items()}, 'calls', dict(calls))
