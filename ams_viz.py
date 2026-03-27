import numpy as np
import pyvista as pv

verts = np.array([
    [-3, -3, -3],
    [ 3,  3, -3],
    [ 3, -3,  3],
    [-3,  3,  3],
    [ 0,  0, -3],
    [ 0, -3,  0],
    [-3,  0,  0],
    [ 3,  0,  0],
    [ 0,  3,  0],
    [ 0,  0,  3],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1,  1,  1],
    [ 0,  0,  0],
], dtype=np.float32)
base_lines = [
    [3, 0, 7, 10],
    [3, 0, 8, 11],
    [3, 0, 9, 12],
    [3, 1, 5, 10],
    [3, 1, 6, 11],
    [3, 1, 9, 13],
    [3, 2, 4, 10],
    [3, 2, 6, 12],
    [3, 2, 8, 13],
    [3, 3, 4, 11],
    [3, 3, 5, 12],
    [3, 3, 7, 13],
    [3, 4, 9, 14],
    [3, 5, 8, 14],
    [3, 6, 7, 14],
]

ams = pv.PolyData(verts, lines=np.hstack(base_lines))
ms1_rows = pv.PolyData(verts, lines=np.hstack([base_lines[x] for x in [0, 8, 9]]))
ms1_cols = pv.PolyData(verts, lines=np.hstack([base_lines[x] for x in [1, 6, 11]]))
# ms2_rows = pv.PolyData(verts, lines=np.hstack([base_lines[x] for x in [0, 4, 13]]))
# ms2_cols = pv.PolyData(verts, lines=np.hstack([base_lines[x] for x in [1, 3, 14]]))

pl = pv.Plotter(off_screen=True)
pl.add_mesh(ams, show_vertices=True, show_edges=True, color="black", render_points_as_spheres=True, point_size=15)
pl.add_mesh(ms1_rows, color="blue", render_lines_as_tubes=True, line_width=5)
pl.add_mesh(ms1_cols, color="red", render_lines_as_tubes=True, line_width=5)

pl.open_gif("pic.gif")

nframe = 120
for theta in np.linspace(0, 360, nframe + 1)[:nframe]:
    pl.camera.azimuth = theta
    pl.write_frame()

pl.close()
