"""New correction/table/model handoff only; no old MRI or training replay."""

import pytest
import torch

from rsna.competitive import paired_logit_control as m


def readout(features, logq):
    values = features[:, :, :12]
    weights = (
        torch.ones_like(values)
        if logq is None
        else torch.exp(-logq)[None, :, None].expand_as(values)
    )
    return (weights * values).sum(1) / weights.sum(1)


def inputs():
    x = torch.arange(2 * 94 * 1024, dtype=torch.float32).reshape(2, 94, 1024) / 1000
    q = torch.linspace(1, 2, 94, dtype=torch.float64)
    q /= q.sum()
    return x, q, m.Recipe(role="influence")


def test_schedule_matches_actual_unchanged_sampler_and_cancels_parent_sampling_noise():
    x, q, recipe = inputs()
    rng = torch.get_rng_state().clone()
    entry = m.schedule(readout, x, "fitting-case", recipe, q)
    assert torch.equal(rng, torch.get_rng_state())
    for epoch in range(3):
        positions, mirror, logq = m.sample("fitting-case", recipe, epoch, q)
        delta = m.lookup(entry, "fitting-case", positions, mirror, logq)
        v = int(mirror)
        actual = readout(x[v : v + 1, positions], logq) + delta
        torch.testing.assert_close(actual, readout(x, None)[v : v + 1], atol=1e-5, rtol=1e-7)
        delta.zero_()
        assert not torch.equal(delta, entry["shift"][epoch])


def test_wrong_windows_or_view_cannot_silently_use_another_epoch():
    x, q, recipe = inputs()
    entry = m.schedule(readout, x, "fitting-case", recipe, q)
    with pytest.raises(ValueError):
        m.lookup(entry, "fitting-case", [93] * 16, False, torch.zeros(16))
    changed = {**entry, "shift": entry["shift"].double()}
    with pytest.raises(ValueError):
        m.lookup(
            changed,
            "fitting-case",
            entry["positions"][0].tolist(),
            bool(entry["mirror"][0]),
            entry["log_proposal"][0],
        )


def test_stochastic_parent_cache_builder_is_refused():
    x, q, recipe = inputs()

    def stochastic(features, logq):
        return readout(features, logq) + torch.rand(features.shape[0], 12)

    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="RNG"):
        m.schedule(stochastic, x, "case", recipe, q)
    assert torch.equal(before, torch.get_rng_state())


class Original(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(12))
        self.calls = 0

    def forward(self, images, log_proposal=None):
        self.calls += 1
        return images * self.weight


class Candidate(m.PairedLogitReadout, Original):
    pass


def test_full_inference_and_parameter_graph_unchanged():
    a = Candidate()
    b = Original()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert torch.equal(a(x), b(x)) and a.calls == 1
    with pytest.raises(ValueError):
        a(x, baseline_shift=torch.zeros(2, 12))


def test_explicit_shift_backpropagates_only_through_actual_student():
    a = Candidate()
    x = torch.ones(2, 12)
    delta = torch.full((2, 12), 0.2)
    z = a(x, torch.zeros(2, 16), baseline_shift=delta)
    z.sum().backward()
    torch.testing.assert_close(z, torch.full((2, 12), 1.2))
    assert torch.equal(a.weight.grad, torch.full((12,), 2.0)) and not delta.requires_grad


@pytest.mark.parametrize(
    "shift",
    [
        None,
        torch.zeros(2, 11),
        torch.zeros(2, 12, dtype=torch.float64),
        torch.full((2, 12), float("nan")),
        torch.zeros(2, 12, requires_grad=True),
    ],
)
def test_invalid_training_control_refused_before_encoder_forward(shift):
    a = Candidate()
    with pytest.raises(ValueError):
        a(torch.ones(2, 12), torch.zeros(2, 16), baseline_shift=shift)
    assert a.calls == 0


def test_identity_and_probability_changes_cannot_reuse_same_image_indexes():
    x, q, recipe = inputs()
    entry = m.schedule(readout, x, "fitting-case", recipe, q)
    pos, view, logq = m.sample("fitting-case", recipe, 0, q)
    with pytest.raises(ValueError):
        m.lookup(entry, "different-study", pos, view, logq)
    with pytest.raises(ValueError):
        m.lookup(entry, "fitting-case", pos, view, logq + 0.01)
