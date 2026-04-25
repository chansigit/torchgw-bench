import textwrap, pathlib
import numpy as np
import io as track_io  # tracks/core/07_cell_morphology/io.py


def test_read_swc_returns_node_array(tmp_path):
    swc = tmp_path / "tiny.swc"
    swc.write_text(textwrap.dedent("""\
        # comment
        1 1 0.0 0.0 0.0 1.0 -1
        2 3 1.0 0.0 0.0 0.5  1
        3 3 1.0 1.0 0.0 0.5  2
        4 3 1.0 1.0 1.0 0.5  3
    """))
    nodes = track_io.read_swc(swc)
    assert nodes.shape == (4, 7)
    assert nodes.dtype == np.float64
    assert nodes[0, 6] == -1
    assert nodes[3, 6] == 3
