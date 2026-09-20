"""Полный эксперимент лабораторной № 1."""
import time
from common import setup,write_json,write_csv,summarize,convergence,save_histories
from model import run


def main():
    cfg,out=setup(); rows=[]; histories={}; best={}
    if cfg['dimension'] != 10 or cfg['lower'] != -600 or cfg['upper'] != 600:
        raise ValueError('Вариант 5 требует d=10 и [-600, 600]')
    for method,scale in [(f'BLX_sigma_{s}',s) for s in cfg['scales']]+[('random',None)]:
        histories[method]=[]
        for seed in range(cfg['seed'],cfg['seed']+cfg['runs']):
            start=time.perf_counter(); value,x,h,calls=run(cfg,seed,scale)
            rows.append(dict(method=method,seed=seed,best=value,evaluations=calls,seconds=time.perf_counter()-start))
            histories[method].append(h)
            if method not in best or value<best[method]['value']: best[method]=dict(value=value,x=x.tolist(),seed=seed)
        print(method, 'finished', flush=True)
    write_csv(out/'runs.csv',rows); write_csv(out/'summary.csv',summarize(rows,['best','seconds','evaluations']))
    write_json(out/'best.json',best); save_histories(out/'histories.csv',histories)
    convergence(out/'convergence.png',histories)
    convergence(out/'convergence_log.png',histories,log=True,title='Сходимость: логарифмическая шкала')

if __name__=='__main__': main()
