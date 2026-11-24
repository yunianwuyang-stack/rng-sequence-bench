from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from ppc_comparator_audit_r22 import frechet_bounds
from rps_missingness_sensitivity_r22 import make_panel, cyclic_rates, tv95, randomize_observed_positions


def test_frechet_bounds_from_marginal_diagnostic_rates():
    assert np.allclose(frechet_bounds([0.9, 0.8, 0.7, 1.0]), [0.4, 0.7])
    assert frechet_bounds([0.0, 0.0, 1.0, 1.0]) == [0.0, 0.0]


def test_rps_long_format_mask_and_bot_pivot():
    rows = []
    for gid in [1, 2]:
        for trial in range(1, 301):
            rows.append({"game_id": gid, "round_index": trial, "is_bot": 0,
                         "player_move": "none" if gid == 1 and trial == 7 else "rock"})
            rows.append({"game_id": gid, "round_index": trial, "is_bot": 1,
                         "player_move": "scissors"})
    df = pd.DataFrame(rows)
    ids, choices, mask, bot = make_panel(df, "game_id", "round_index", "player_move",
                                         bot_col=None, is_bot_col="is_bot", bot_flag="1")
    assert len(ids) == 2 and choices.shape == (2, 300) and mask.shape == (2, 300)
    assert not mask[0, 6]
    assert mask[1].all()
    assert bot[0, 0] == 2
    assert tv95(choices, mask)[2] == 2
    assert cyclic_rates(choices, mask)["n_transitions"] == int((mask[:, 1:] & mask[:, :-1]).sum())


def test_randomized_missing_positions_preserve_observed_move_order():
    y = np.array([[0, 2, 1, 0, 2]], dtype=np.int8)
    m = np.array([[True, False, True, True, False]])
    yr, mr = randomize_observed_positions(y, m, np.random.default_rng(9))
    assert sorted(yr[mr].tolist()) == sorted(y[m].tolist())
    assert yr[mr].tolist() == y[m].tolist()
    assert int(mr.sum()) == int(m.sum())
