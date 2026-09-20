import json
import unittest
import numpy as np
from common import ROOT
from model import objective,reflect,run

class TestGriewank(unittest.TestCase):
    def test_known_minimum_and_nonnegative(self):
        self.assertEqual(float(objective(np.zeros(10))),0.)
        self.assertTrue(np.all(objective(np.random.default_rng(7).uniform(-600,600,(100,10)))>=0))
    def test_reflection_many_widths(self):
        x=reflect(np.array([-6001.,-601.,0.,601.,6001.]),-600,600)
        self.assertTrue(np.all((x>=-600)&(x<=600)))
        self.assertEqual(x[1],-599); self.assertEqual(x[3],599)
    def test_budget_elitism_and_reproducibility(self):
        cfg=json.loads((ROOT/'config.json').read_text()); cfg['generations']=3
        a=run(cfg,9,.01); b=run(cfg,9,.01); c=run(cfg,9,None)
        self.assertEqual(a[3],cfg['population']*4); self.assertEqual(c[3],a[3])
        np.testing.assert_array_equal(a[1],b[1]); self.assertTrue(np.all(np.diff(a[2])<=0))
        self.assertAlmostEqual(a[0],float(objective(a[1])))
if __name__=='__main__': unittest.main()
