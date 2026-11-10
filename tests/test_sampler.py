"""Coupled sampling keeps each agent's policy and raises agreement."""
import numpy as np

from padp.marl.sampler import NoiseSource, coalition_stats, gumbel_argmax

DRAWS = 200_000


def _softmax(logits):
    z = np.exp(logits - logits.max())
    return z / z.sum()


def _draw(logits_a, logits_b, shared: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    g_a = rng.gumbel(size=(DRAWS, len(logits_a)))
    g_b = g_a if shared else rng.gumbel(size=(DRAWS, len(logits_b)))
    return gumbel_argmax(np.broadcast_to(logits_a, g_a.shape), g_a), gumbel_argmax(np.broadcast_to(logits_b, g_b.shape), g_b)


def test_shared_noise_preserves_each_marginal():
    logits_a, logits_b = np.array([1.0, 0.2, -0.5, 0.0, 0.7]), np.array([0.1, 1.5, 0.0, -1.0, 0.3])
    a, b = _draw(logits_a, logits_b, shared=True)
    np.testing.assert_allclose(np.bincount(a, minlength=5) / DRAWS, _softmax(logits_a), atol=0.01)
    np.testing.assert_allclose(np.bincount(b, minlength=5) / DRAWS, _softmax(logits_b), atol=0.01)


def test_identical_policies_always_agree_when_coupled():
    logits = np.zeros(5)  # uniform: the worst case for independent sampling
    a, b = _draw(logits, logits, shared=True)
    assert np.all(a == b)


def test_independent_agreement_matches_theory():
    logits_a, logits_b = np.array([1.0, 0.2, -0.5, 0.0, 0.7]), np.array([0.1, 1.5, 0.0, -1.0, 0.3])
    a, b = _draw(logits_a, logits_b, shared=False)
    expected = float(np.dot(_softmax(logits_a), _softmax(logits_b)))
    assert abs(np.mean(a == b) - expected) < 0.01


def test_coupled_agreement_beats_bound():
    """Gumbel coupling agrees with probability at least (1 - TV) / (1 + TV) (Daliri et al.)."""
    logits_a, logits_b = np.array([1.0, 0.2, -0.5, 0.0, 0.7]), np.array([0.1, 1.5, 0.0, -1.0, 0.3])
    a, b = _draw(logits_a, logits_b, shared=True)
    tv = 0.5 * np.abs(_softmax(logits_a) - _softmax(logits_b)).sum()
    assert np.mean(a == b) >= (1 - tv) / (1 + tv) - 0.01


def test_noise_source_shares_only_within_an_anchor():
    source = NoiseSource("coupled", run_seed=3, head_sizes=(5, 3))
    anchors = np.array([[0, 0], [0, 0], [1, 1], [-1, -1]])
    move, interact = source.sample(anchors, env_idx=0, episode_idx=2, t=7)
    np.testing.assert_array_equal(move[0], move[1])
    np.testing.assert_array_equal(interact[0], interact[1])
    assert not np.allclose(move[0], move[2]) and not np.allclose(move[0], move[3])
    # Any member can rebuild the shared noise on its own: it depends only on common knowledge.
    np.testing.assert_array_equal(move[0], source.shared(0, 2, 0, 7, 0))


def test_independent_noise_never_shares():
    source = NoiseSource("independent", run_seed=3, head_sizes=(5, 3))
    move, _ = source.sample(np.zeros((2, 2), dtype=int), env_idx=0, episode_idx=0, t=0)
    assert not np.allclose(move[0], move[1])


def test_coalition_stats():
    probs = np.array([[0.5, 0.5, 0, 0, 0], [0.5, 0.5, 0, 0, 0], [1, 0, 0, 0, 0]])
    anchors = np.array([[0, 0], [0, 0], [-1, -1]])
    q, tv, count = coalition_stats(probs, anchors)
    assert count == 1 and abs(q - 0.5) < 1e-9 and abs(tv) < 1e-9


ANCHORS = np.array([[0, 0], [0, 0], [1, 1], [-1, -1]])


def test_global_sampler_shares_noise_across_all_agents():
    move, interact = NoiseSource("global", run_seed=3, head_sizes=(5, 3)).sample(ANCHORS, 0, 2, 7)
    assert all(np.array_equal(move[0], m) for m in move) and all(np.array_equal(interact[0], x) for x in interact)


def test_movement_only_coupling_keeps_interaction_noise_private():
    move, interact = NoiseSource("coupled_move", run_seed=3, head_sizes=(5, 3)).sample(ANCHORS, 0, 2, 7)
    np.testing.assert_array_equal(move[0], move[1])
    assert not np.allclose(interact[0], interact[1])


def test_anchor_error_zero_is_exactly_coupled_and_one_breaks_all_sharing():
    plain = NoiseSource("coupled", run_seed=3, head_sizes=(5, 3))
    zero = NoiseSource("coupled", run_seed=3, head_sizes=(5, 3), anchor_error=0.0)
    for t in range(5):
        for x, y in zip(plain.sample(ANCHORS, 0, 2, t), zero.sample(ANCHORS, 0, 2, t)):
            np.testing.assert_array_equal(x, y)
    move, interact = NoiseSource("coupled", run_seed=3, head_sizes=(5, 3), anchor_error=1.0).sample(ANCHORS, 0, 2, 7)
    assert not np.allclose(move[0], move[1]) and not np.allclose(interact[0], interact[1])
