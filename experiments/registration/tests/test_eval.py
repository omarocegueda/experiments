import numpy as np
from experiments.registration.evaluation import compute_jaccard
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)

def test_compute_jaccard():
    sh = (20, 25, 30)
    S = np.array(range(sh[0]*sh[1]*sh[2]), dtype=np.int32).reshape(sh)
    A = S % 2
    B = (S + 1) % 2

    expected = np.array([0.0, 0.0])
    actual = np.array(compute_jaccard(A, B))
    assert_array_almost_equal(actual, expected)

    expected = np.array([1.0, 1.0])
    actual = np.array(compute_jaccard(A, A))
    assert_array_almost_equal(actual, expected)

    B = S % 3
    expected = np.array([0.25, 0.25, 0.0])
    actual = np.array(compute_jaccard(A, B))



if __name__=="__main__":
    test_compute_jaccard()
