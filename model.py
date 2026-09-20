"""Гриванк и вещественный ГА; готовые оптимизаторы не используются."""
import numpy as np


def objective(x):
    x = np.asarray(x, dtype=float)
    return 1 + np.sum(x*x, axis=-1)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,x.shape[-1]+1))), axis=-1)


def reflect(x, lower, upper):
    """Отражение работает и при выходе более чем на ширину области."""
    width = upper-lower
    return lower + width - np.abs((x-lower) % (2*width)-width)


def run(cfg, seed, sigma=None):
    rng=np.random.default_rng(seed)
    n,d=cfg['population'],cfg['dimension']
    lo,hi=cfg['lower'],cfg['upper']
    pop=rng.uniform(lo,hi,(n,d)); scores=objective(pop)
    history=[float(scores.min())]; calls=n
    best=pop[scores.argmin()].copy(); best_score=float(scores.min())
    for _ in range(cfg['generations']):
        if sigma is None:
            children=rng.uniform(lo,hi,(n,d))
        else:
            ids=rng.integers(n,size=(2,n,3))
            winners=ids[np.arange(2)[:,None],np.arange(n)[None,:],scores[ids].argmin(axis=2)]
            a,b=pop[winners[0]],pop[winners[1]]
            # BLX-alpha = 0.3: независимо по каждой координате.
            weight=rng.uniform(-.3,1.3,(n,d))
            children=np.where(rng.random((n,1))<cfg['crossover_probability'],a+weight*(b-a),a)
            children += (rng.random((n,d))<cfg['mutation_probability'])*rng.normal(0,sigma*(hi-lo),(n,d))
            children=reflect(children,lo,hi)
        values=objective(children); calls+=n
        if values.min()<best_score:
            best_score=float(values.min()); best=children[values.argmin()].copy()
        if sigma is not None:
            merged=np.concatenate([pop,children]); all_scores=np.r_[scores,values]
            order=np.argsort(all_scores,kind='stable')[:n]
            pop,scores=merged[order],all_scores[order]
        history.append(best_score)
    return best_score,best,history,calls
