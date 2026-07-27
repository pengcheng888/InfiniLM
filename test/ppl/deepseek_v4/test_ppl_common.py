from ppl_common import iter_windows


def test_windows_score_each_target_once():
    windows = list(iter_windows(range(20), 8, 4, None))
    scored = []
    for window in windows:
        scored.extend(range(window.score_start, window.score_end))
    assert scored == list(range(1, 20))


def test_max_scored_tokens():
    windows = list(iter_windows(range(20), 8, 4, 7))
    assert sum(window.scored_token_count for window in windows) == 7
