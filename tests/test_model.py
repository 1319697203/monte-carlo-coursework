"""针对科学结论所依赖的核心计算做小规模、可手算的检验。"""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from simulate import schedule, sample_durations, pert_parameters

class ModelTests(unittest.TestCase):
    def test_parallel_merge_and_path_switch(self):
        # A之后B/C并行，再D汇合。两轮最长分支互换，总工期均为8。
        pred=[[],[0],[0],[1,2]]; succ=[[1,2],[3],[3],[]]
        result=schedule([[2,5,3,1],[2,3,5,1]],pred,succ)
        np.testing.assert_allclose(result[0],[8,8])
        np.testing.assert_allclose(result[1][:,3],[7,7])
        np.testing.assert_array_equal(result[-1],[[1,1,0,1],[1,0,1,1]])

    def test_two_sinks_and_tied_paths(self):
        r=schedule([[2,3,3]], [[],[0],[0]], [[1,2],[],[]])
        self.assertEqual(r[0][0],5)
        self.assertTrue(r[-1].all())

    def test_pert_mean_and_boundaries(self):
        tasks=[{'a':2.,'m':4.,'b':10.},{'a':3.,'m':3.,'b':3.}]
        draw=sample_durations(tasks,100000,17)
        self.assertTrue(np.all((draw[:,0]>=2)&(draw[:,0]<=10)))
        self.assertTrue(np.all(draw[:,1]==3))
        self.assertAlmostEqual(draw[:,0].mean(),(2+4*4+10)/6,delta=.025)
        a,b=pert_parameters(np.array([2]),np.array([4]),np.array([10]))
        np.testing.assert_allclose([a[0],b[0]],[2,4])

if __name__=='__main__': unittest.main()
