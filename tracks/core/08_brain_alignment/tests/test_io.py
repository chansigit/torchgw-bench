import numpy as np
import io_brain


def test_load_fsaverage_mesh_returns_vertices_and_faces():
    verts, faces = io_brain.load_fsaverage_mesh("fsaverage5", hemi="left")
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert verts.shape[0] == 10242  # fsaverage5 left hemi vertex count
    assert faces.dtype == np.int64


def test_load_localizer_contrast_volume_returns_3d():
    """Probe the on-disk Localizer to check 1 contrast loads as 3D NIfTI."""
    img = io_brain.load_localizer_contrast_volume(
        "S01", "left button press (auditory cue)")
    # MNI152 contrast maps are 3D
    assert img.ndim == 3
    assert img.shape[0] >= 50  # rough lower bound on a brain volume axis
