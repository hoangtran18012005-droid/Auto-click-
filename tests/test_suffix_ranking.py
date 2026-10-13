import numpy as np
from bot.compiled_planner import kernel


def test_common_prefix_does_not_change_suffix_ranking_or_hide_collision():
    gates = np.array([[180., 266., 110., 230.], [375., 461., 60., 180.]])
    args = (170., 100., 710.38, -248.58494, 139.6, gates, 145., 26., 25., 0., 356.)
    tail = (.04, 1/30, 70, .18, .25, 200, 2., 8., -1, True)
    action, margin = kernel(*args, 1e6, *tail)
    for prefix in (0., -20.):
        limited_action, limited_margin = kernel(*args, prefix, *tail)
        assert limited_action == action
        assert limited_margin == min(prefix, margin)
