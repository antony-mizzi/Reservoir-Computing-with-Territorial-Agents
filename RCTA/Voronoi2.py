import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Shoelace area
# ------------------------------------------------------------
def polygon_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) -
                        np.dot(y, np.roll(x, -1)))

# ------------------------------------------------------------
# Half-plane clipping of a convex polygon
# ------------------------------------------------------------
def clip_with_halfplane(poly, p, n):
    if len(poly) == 0:
        return poly

    new_poly = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da = np.dot(a - p, n)
        db = np.dot(b - p, n)

        if da <= 0:
            new_poly.append(a)

        if da * db < 0:  # edge crosses boundary
            t = da / (da - db)
            new_poly.append(a + t * (b - a))

    return np.array(new_poly)


# ------------------------------------------------------------
# Compute clipped Voronoi cell for site i
# ------------------------------------------------------------
def voronoi_cell(points, i, a, b):
    # Start with the bounding box polygon
    poly = np.array([
        [-a, -b],
        [ a, -b],
        [ a,  b],
        [-a,  b]
    ], dtype=float)

    pi = points[i]

    for j in range(len(points)):
        if j == i:
            continue

        pj = points[j]

        # Midpoint of pi and pj
        m = 0.5 * (pi + pj)

        # Normal vector pointing TOWARD pj (so outward from pi's region)
        n = pj - pi

        # Clip polygon by half-plane
        poly = clip_with_halfplane(poly, m, n)

        if len(poly) == 0:
            break

    return poly


# ------------------------------------------------------------
# Compute all clipped Voronoi polygons
# ------------------------------------------------------------
def clipped_voronoi_polygons(points, a, b):
    polys = []
    for i in range(len(points)):
        poly = voronoi_cell(points, i, a, b)
        polys.append(poly)
    return polys

# ------------------------------------------------------------
# Plot Voronoi tessellation colored by clipped area
# ------------------------------------------------------------
def plot_voronoi_clipped(points, a, b):
    polys = clipped_voronoi_polygons(points, a, b)
    areas = np.array([polygon_area(p) if len(p) >= 3 else 0 for p in polys])
    print(areas)
    vmin, vmax = areas.min(), areas.max()

    fig, ax = plt.subplots(figsize=(7,7))

    for poly, area in zip(polys, areas):
        color = plt.cm.Greys((area - vmin) / (vmax - vmin))
        ax.fill(poly[:,0], poly[:,1], color=color, edgecolor='black', linewidth=0.5, linestyle="--", alpha=0.3)

    ax.plot(points[:,0], points[:,1], 'x', color='k', markersize=5)
    ax.set_xlim(-a, a)
    ax.set_ylim(-b, b)
    ax.set_aspect('equal')

    plt.colorbar(plt.cm.ScalarMappable(cmap='Greys'),
                 ax=ax, label="Clipped Voronoi Cell Area")
    plt.title("Voronoi Tessellation Colored by Area (Clipped to Box)")
    plt.show()
    

def cell_area_at_coords(points,a=3,b=3,n=121):
    xs = np.linspace(-a, a, n)
    ys = np.linspace(-b, b, n)
    grid = np.array([(x, y) for x in xs for y in ys])

    A = np.zeros(len(grid))
    polys = clipped_voronoi_polygons(points, a, b)
    areas = np.array([polygon_area(p) for p in polys])

    # nearest site for each grid point
    d2 = np.sum((grid[:,None,:] - points[None,:,:])**2, axis=2)
    nearest = np.argmin(d2, axis=1)

    # accumulate areas
    A += areas[nearest]
    return xs, ys, A.reshape(n, n)
    
    



