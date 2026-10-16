import pytest

from rsna.competitive.bounded_preparation import bounded_results


class Pool:
    def __init__(self):
        self.outstanding = self.maximum = self.submissions = 0

    def apply_async(self, function, args):
        self.outstanding += 1
        self.submissions += 1
        self.maximum = max(self.maximum, self.outstanding)
        pool = self

        class Result:
            def get(self):
                pool.outstanding -= 1
                return function(*args)

        return Result()


def test_slow_consumer_is_bounded_without_losing_or_duplicating_tasks():
    p = Pool()
    values = bounded_results(p, lambda x: x * 2, range(10000), limit=4)
    assert next(values) == 0
    assert p.submissions == 4
    assert p.outstanding == 3
    assert next(values) == 2
    assert p.submissions == 5
    assert list(values) == list(range(4, 20000, 2))
    assert p.maximum == 4 and p.submissions == 10000 and p.outstanding == 0


@pytest.mark.parametrize("n", [0, 1, 3, 4, 5])
def test_small_input_and_invalid_limit(n):
    p = Pool()
    assert list(bounded_results(p, lambda x: x, range(n))) == list(range(n))
    assert p.maximum <= 4
    with pytest.raises(ValueError):
        list(bounded_results(p, lambda x: x, range(n), limit=0))


def test_worker_error_is_propagated_and_never_automatically_retried():
    p = Pool()

    def fail(x):
        if x == 2:
            raise RuntimeError("original worker")
        return x

    v = bounded_results(p, fail, range(10), limit=2)
    assert next(v) == 0 and next(v) == 1
    with pytest.raises(RuntimeError, match="original worker"):
        next(v)
    assert p.submissions == 4
