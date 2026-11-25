import numpy as np 
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Shoelace formula
# ------------------------------------------------------------
def polygon_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) -
                        np.dot(y, np.roll(x, -1)))

# ------------------------------------------------------------
# Sutherland–Hodgman polygon clipping to rectangle
# ------------------------------------------------------------
def clip_polygon(poly, xmin, xmax, ymin, ymax, tol=1e-12):
    def clip(poly, inside, intersect):
        out = []
        for p1, p2 in zip(poly, np.roll(poly, -1, axis=0)):
            i1, i2 = inside(p1), inside(p2)
            if i1 and i2:
                out.append(p2)
            elif i1 and not i2:
                out.append(intersect(p1, p2))
            elif not i1 and i2:
                out.append(intersect(p1, p2))
                out.append(p2)
        return np.array(out)

    def intersect_x(p1, p2, x):
        t = (x - p1[0]) / (p2[0] - p1[0])
        return np.array([x, p1[1] + t*(p2[1] - p1[1])])

    def intersect_y(p1, p2, y):
        t = (y - p1[1]) / (p2[1] - p1[1])
        return np.array([p1[0] + t*(p2[0] - p1[0]), y])

    poly = clip(poly, lambda p: p[0] >= xmin, lambda p1,p2: intersect_x(p1,p2,xmin))
    poly = clip(poly, lambda p: p[0] <= xmax, lambda p1,p2: intersect_x(p1,p2,xmax))
    poly = clip(poly, lambda p: p[1] >= ymin, lambda p1,p2: intersect_y(p1,p2,ymin))
    poly = clip(poly, lambda p: p[1] <= ymax, lambda p1,p2: intersect_y(p1,p2,ymax))
    
    return poly
        

def point_in_poly(pt, poly):
    x, y = pt
    xs, ys = poly[:,0], poly[:,1]
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def add_owned_corners(poly, i, points, a, b):
    if len(poly) == 0:
        return poly

    corners = np.array([[ a,  b],
                        [ a, -b],
                        [-a,  b],
                        [-a, -b]])

    # For each corner, check which site it belongs to
    extra = []
    for c in corners:
        d2 = np.sum((points - c)**2, axis=1)
        owner = np.argmin(d2)
        if owner == i:
            extra.append(c)

    if extra:
        poly = np.vstack([poly, np.vstack(extra)])
        poly = sort_polygon(poly)

    return poly


def convex_hull(points):
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points

    pts = points[np.lexsort((points[:,1], points[:,0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1])
    return hull

# ------------------------------------------------------------
# Build finite polygons for all Voronoi regions (including infinite)
# ------------------------------------------------------------
def clipped_voronoi_polygons(points, a, b):
    vor = Voronoi(points)
    xmin, xmax = -a, a
    ymin, ymax = -b, b

    new_polys = []
    center = points.mean(axis=0)
    radius = 10 * max(a, b)

    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        # Finite region
        if -1 not in region:
            poly = vor.vertices[region]
            poly = clip_polygon(poly, xmin, xmax, ymin, ymax)
            poly = dedupe(poly)
            new_polys.append(poly)
            continue

        # Infinite region → reconstruct
        
        region_vertices = []
        for (p, q), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            
            if p == i or q == i:
                if v1 >= 0:
                    region_vertices.append(v1)
                if v2 >= 0:
                    region_vertices.append(v2)
                
                if v1 == -1 or v2 == -1:
                    
                    p1, p2 = points[[p, q]]
                    tangent = p2 - p1
                    normal = np.array([-tangent[1], tangent[0]])
                    normal /= np.linalg.norm(normal)

                    midpoint = (p1 + p2) / 2
                    direction = np.sign(np.dot(midpoint - center, normal)) * normal

                    far_point = ray_box_intersection(midpoint, direction, xmin, xmax, ymin, ymax)
                    vor.vertices = np.vstack([vor.vertices, far_point])
                    region_vertices.append(len(vor.vertices) - 1)
        

        #original_poly = dedupe(original_poly)
        poly = vor.vertices[region_vertices]
        poly = dedupe(poly)
        poly = convex_hull(poly)
        
        poly = clip_polygon(poly, xmin, xmax, ymin, ymax)
        poly = dedupe(poly)
        poly = add_owned_corners(poly, i, points, a, b)
        poly = dedupe(poly)
        poly = convex_hull(poly)
        new_polys.append(poly)

    return new_polys

def ray_box_intersection(p, d, xmin, xmax, ymin, ymax):
    """
    p = starting point of ray
    d = direction (unit vector)
    returns intersection point with the box boundary
    """
    t_vals = []

    # Avoid division by zero
    if d[0] != 0:
        t_vals += [(xmin - p[0]) / d[0], (xmax - p[0]) / d[0]]
    if d[1] != 0:
        t_vals += [(ymin - p[1]) / d[1], (ymax - p[1]) / d[1]]

    # Only positive t (forward along ray)
    t_vals = [t for t in t_vals if t > 0]

    # Compute intersection points
    pts = [p + t * d for t in t_vals]

    # Keep only points actually on the box boundary
    pts = [pt for pt in pts if
           xmin - 1e-12 <= pt[0] <= xmax + 1e-12 and
           ymin - 1e-12 <= pt[1] <= ymax + 1e-12]

    # Choose the closest intersection
    return min(pts, key=lambda q: np.linalg.norm(q - p))

def dedupe(poly, tol=1e-12):
    out = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < tol:
        out.pop()
    return np.array(out)

def sort_polygon(poly):
    c = poly.mean(axis=0)
    angles = np.arctan2(poly[:,1] - c[1], poly[:,0] - c[0])
    return poly[np.argsort(angles)]

# ------------------------------------------------------------
# Plot Voronoi tessellation colored by clipped area
# ------------------------------------------------------------
def plot_voronoi_clipped(points, a, b):
    polys = clipped_voronoi_polygons(points, a, b)
    print(polys)
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

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------
points = np.random.normal(0,1,size=(8,2))
print(points)
plot_voronoi_clipped(points, a=3, b=3)
