#pragma once

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <set>

#ifdef HAVE_EIGEN
#   include <Eigen/Core>
#endif

namespace delaunator {

constexpr double EPSILON = std::numeric_limits<double>::epsilon();
constexpr std::size_t INVALID_INDEX = std::numeric_limits<std::size_t>::max();

class Delaunator {

public:
    inline Delaunator(std::vector<double>& in_coords)
    {
        if(in_coords.size() % 2 != 0) {
            throw std::invalid_argument("Argument must have an even number of coorinates");
        }
        triangulate(in_coords.data(), in_coords.size() / 2);
    }

    inline Delaunator(double* coords, size_t num_points)
    {
        triangulate(coords, num_points);
    }

#ifdef HAVE_EIGEN
    inline Delaunator(Eigen::Matrix<double,2,Eigen::Dynamic>& points)
    {
        triangulate(points.data(), points.cols());
    }
    inline Delaunator(std::vector<Eigen::Vector2d>& points)
    {
        triangulate((double*)points.data(), points.size());
    }
#endif // HAVE_EIGEN

    double get_hull_area();

    inline size_t get_num_points() const
    {
        return num_points;
    }

    inline size_t get_num_halfedges() const
    {
        return triangles.size();
    }

    inline size_t get_num_edges() const
    {
        return triangles.size() / 2;
    }

    inline size_t get_num_triangles() const
    {
        return triangles.size() / 3;
    }

    // Triangle index, t \in [0, get_num_triangles()-1]
    // Halfedge index, e \in [0,1,2]
    // Returns halfedge index \in [0, get_num_halfedges()-1]
    static inline size_t edge_of_triangle(size_t t, size_t e) {
        return 3 * t + e;
    }

    // Halfedge index, e \in [0, get_num_halfedges()-1]
    // Returns triangle index \in [0, get_num_triangles()-1]
    static inline size_t triangle_of_edge(size_t e)  {
        return e / 3;
    }

    // Halfedge index, e \in [0, get_num_halfedges()-1]
    // Returns haldedge index \in [0, get_num_halfedges()-1]
    static inline size_t next_halfedge(size_t e) {
        return (e % 3 == 2) ? (e - 2) : (e + 1);
    }

    // e \in [0, get_num_halfedges()-1]
    // Returns haldedge index \in [0, get_num_halfedges()-1]
    static inline size_t prev_halfedge(size_t e) {
        return (e % 3 == 0) ? (e + 2) : (e - 1);
    }

    // e \in [0, get_num_halfedges()-1]
    // vi \in [0,1]
    inline size_t get_point_of_edge(size_t e, size_t vi) const
    {
        return triangles[vi == 0 ? e : next_halfedge(e)];
    }

    // t \in [0, get_num_triangles()-1]
    // vi \in [0,1,2]
    inline size_t get_point_of_triangle(size_t t, size_t vi) const
    {
        return triangles[edge_of_triangle(t, vi)];
    }

    // e \in [0, get_num_halfedges()-1]
    // Returns point index \in [0, get_total_num_points()-1]
    inline size_t get_point_opposite_of_halfedge(size_t e) const
    {
        return triangles[prev_halfedge(e)];
    }

    // e \in [0, get_num_halfedges()-1]
    // Returns point index \in [0, get_total_num_points()-1]
    inline size_t get_point_at_halfedge_start(size_t e) const
    {
        return triangles[e];
    }

    // e \in [0, get_num_halfedges()-1]
    // Returns point index \in [0, get_total_num_points()-1]
    inline size_t get_point_at_halfedge_end(size_t e) const
    {
        return triangles[next_halfedge(e)];
    }

    // e \in [0, get_num_halfedges()-1]
    // Returns halfedge index opposite to e, or INVALID_INDEX iff
    // this half-edge lies on a boundary
    inline size_t get_opposite_halfedge(size_t e) const
    {
        return halfedges[e];
    }

#ifdef HAVE_EIGEN
    // Return the coordinates of point with index pi
    inline Eigen::Vector2d get_point(size_t pi) const
    {
        return Eigen::Vector2d(coords[2*pi+0], coords[2*pi+1]);
    }

    // Return the point indices that make up halfedge ei
    inline Eigen::Vector2i get_halfedge(size_t ei) const
    {
        return Eigen::Vector2i(
            triangles[ei],
            triangles[next_halfedge(ei)]
        );
    }

    // Return the point indices that make up triangle ti
    inline Eigen::Vector3i get_triangle(size_t ti) const
    {
        return Eigen::Vector3i(
            get_point_of_triangle(ti, 0),
            get_point_of_triangle(ti, 1),
            get_point_of_triangle(ti, 2)
        );
    }

    // Return the point indices that make up quad defined by
    // one of the interior halfedges ei
    // Points are defined in ring around perimeter starting.
    inline Eigen::Vector4i get_quad(size_t e) const
    {
        size_t oppe = get_opposite_halfedge(e);
        return Eigen::Vector4i(
            get_point_at_halfedge_start(e),
            get_point_opposite_of_halfedge(e),
            get_point_at_halfedge_start(oppe),
            get_point_opposite_of_halfedge(oppe)
        );
    }
#endif

    // Edge Visitation
    // Will call func(index, x, y)
    template<typename Func>
    void for_each_point(Func func) {
        for (size_t v = 0; v < get_num_points(); v++) {
            func(v, coords[2*v + 0], coords[2*v + 1]);
        }
    }

    // Edge Visitation
    // Will call func(edge_index, point1_index, point2_index)
    template<typename Func>
    void for_each_edge(Func func) {
        for (size_t e = 0; e < get_num_halfedges(); e++) {
            // TODO: Will this miss edges on the convex hull since halfedges[e] == INVALID_INDEX
            if (e > halfedges[e]) {
                const size_t p = triangles[e];
                const size_t q = triangles[next_halfedge(e)];
                func(e, p, q);
            }
        }
    }

    // Triangle Visitation
    // Will call func(triangle_index, point1_index, point2_index, point3_index)
    template<typename Func>
    void for_each_triangle(Func func) {
        for(size_t t = 0; t < get_num_triangles(); t++) {
            func( t,
                get_point_of_triangle(t, 0),
                get_point_of_triangle(t, 1),
                get_point_of_triangle(t, 2)
            );
        }
    }

    // Quad Visitation - for each pair of adjacent triangles
    // Will call func(triangle1_index, triangle2_index, point1_index, point2_index, point3_index, point4_index)
    template<typename Func>
    void for_each_quad(Func func) {
        // We possibly have a quad for every edge
        // halfedges[e] == -1 for edges on the hull
        for (size_t e = 0; e < triangles.size(); e++) {
            const size_t oppe = get_opposite_halfedge(e);
            if (e > oppe && oppe != INVALID_INDEX) {
//                // Get the triangles adjacent to e
//                size_t t1 = triangle_of_edge(e);
//                size_t t2 = triangle_of_edge(oppe);
//                size_t p1 = get_point_at_halfedge_start(e);
//                size_t p2 = get_point_opposite_of_halfedge(e);
//                size_t p3 = get_point_at_halfedge_start(oppe);
//                size_t p4 = get_point_opposite_of_halfedge(oppe);
//                func(e, t1,t2,p1,p2,p3,p4);
                func(e);
            }
        }
    }

    // Quad is specified by its interior edge
    template<typename Func>
    void for_each_adjacent_vertex_of_edge_start(size_t e, Func func) {
        size_t curr_e = e;
        func(get_point_at_halfedge_end(curr_e));

        while(true){
            curr_e = get_opposite_halfedge(next_halfedge(curr_e));
            if(curr_e != e && curr_e != INVALID_INDEX) {
                func(get_point_at_halfedge_end(curr_e));
            }else{
                break;
            }
        }
    }

    // Quad is specified by one of its two interior halfedges
    // method will call func(v) for each vertex adjacent to this quad.
    template<typename Func>
    void for_each_point_adjacent_quad(size_t e, Func func) {
        std::set<size_t> pts;

        auto add_pt = [&pts](size_t i){pts.insert(i);};

        const size_t oppe = get_opposite_halfedge(e);
        if(oppe == INVALID_INDEX) {
            throw std::invalid_argument("Not a quad");
        }
        const size_t nxte = next_halfedge(e);
        const size_t nxtoppe = next_halfedge(oppe);
        for_each_adjacent_vertex_of_edge_start(nxte, add_pt);
        for_each_adjacent_vertex_of_edge_start(prev_halfedge(e), add_pt);
        for_each_adjacent_vertex_of_edge_start(nxtoppe, add_pt);
        for_each_adjacent_vertex_of_edge_start(prev_halfedge(oppe), add_pt);

        // Now remove quad points
        pts.erase(get_point_at_halfedge_start(e));
        pts.erase(get_point_at_halfedge_start(oppe));
        pts.erase(get_point_at_halfedge_end(nxte));
        pts.erase(get_point_at_halfedge_end(nxtoppe));

        for(const auto& v : pts) {
            func(v);
        }
    }


    // Expose raw access to arrays (be carefull!)
    double* coords;
    size_t num_points;

    // triangles[e] is the point id where the half-edge starts
    std::vector<std::size_t> triangles;

    // halfedges[e] is the opposite half-edge in the adjacent triangle,
    // or INVALID_INDEX if there is no adjacent triangle
    std::vector<std::size_t> halfedges;

    //
    std::vector<std::size_t> hull_prev;
    std::vector<std::size_t> hull_next;
    std::vector<std::size_t> hull_tri;
    std::size_t hull_start;

private:
    void triangulate(double *points, size_t num_points);

    std::vector<std::size_t> m_hash;
    double m_center_x;
    double m_center_y;
    std::size_t m_hash_size;
    std::vector<std::size_t> m_edge_stack;

    std::size_t legalize(std::size_t a);
    std::size_t hash_key(double x, double y) const;
    std::size_t add_triangle(
        std::size_t i0,
        std::size_t i1,
        std::size_t i2,
        std::size_t a,
        std::size_t b,
        std::size_t c);
    void link(std::size_t a, std::size_t b);
};

/////////////////////////////////////////////////////////////////
/// Implementation
/////////////////////////////////////////////////////////////////

//@see https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
inline size_t fast_mod(const size_t i, const size_t c) {
    return i >= c ? i % c : i;
}

// Kahan and Babuska summation, Neumaier variant; accumulates less FP error
inline double sum(const std::vector<double>& x) {
    double sum = x[0];
    double err = 0.0;

    for (size_t i = 1; i < x.size(); i++) {
        const double k = x[i];
        const double m = sum + k;
        err += std::fabs(sum) >= std::fabs(k) ? sum - m + k : k - m + sum;
        sum = m;
    }
    return sum + err;
}

inline double dist(
    const double ax,
    const double ay,
    const double bx,
    const double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

inline double circumradius(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double cx,
    const double cy) {
    const double dx = bx - ax;
    const double dy = by - ay;
    const double ex = cx - ax;
    const double ey = cy - ay;

    const double bl = dx * dx + dy * dy;
    const double cl = ex * ex + ey * ey;
    const double d = dx * ey - dy * ex;

    const double x = (ey * bl - dy * cl) * 0.5 / d;
    const double y = (dx * cl - ex * bl) * 0.5 / d;

    if ((bl > 0.0 || bl < 0.0) && (cl > 0.0 || cl < 0.0) && (d > 0.0 || d < 0.0)) {
        return x * x + y * y;
    } else {
        return std::numeric_limits<double>::max();
    }
}

inline bool orient(
    const double px,
    const double py,
    const double qx,
    const double qy,
    const double rx,
    const double ry) {
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;
}

inline std::pair<double, double> circumcenter(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double cx,
    const double cy) {
    const double dx = bx - ax;
    const double dy = by - ay;
    const double ex = cx - ax;
    const double ey = cy - ay;

    const double bl = dx * dx + dy * dy;
    const double cl = ex * ex + ey * ey;
    const double d = dx * ey - dy * ex;

    const double x = ax + (ey * bl - dy * cl) * 0.5 / d;
    const double y = ay + (dx * cl - ex * bl) * 0.5 / d;

    return std::make_pair(x, y);
}

struct compare {

    double* coords;
    double cx;
    double cy;

    bool operator()(std::size_t i, std::size_t j) {
        const double d1 = dist(coords[2 * i], coords[2 * i + 1], cx, cy);
        const double d2 = dist(coords[2 * j], coords[2 * j + 1], cx, cy);
        const double diff1 = d1 - d2;
        const double diff2 = coords[2 * i] - coords[2 * j];
        const double diff3 = coords[2 * i + 1] - coords[2 * j + 1];

        if (diff1 > 0.0 || diff1 < 0.0) {
            return diff1 < 0;
        } else if (diff2 > 0.0 || diff2 < 0.0) {
            return diff2 < 0;
        } else {
            return diff3 < 0;
        }
    }
};

inline bool in_circle(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double cx,
    const double cy,
    const double px,
    const double py) {
    const double dx = ax - px;
    const double dy = ay - py;
    const double ex = bx - px;
    const double ey = by - py;
    const double fx = cx - px;
    const double fy = cy - py;

    const double ap = dx * dx + dy * dy;
    const double bp = ex * ex + ey * ey;
    const double cp = fx * fx + fy * fy;

    return (dx * (ey * cp - bp * fy) -
            dy * (ex * cp - bp * fx) +
            ap * (ex * fy - ey * fx)) < 0.0;
}

inline bool check_pts_equal(double x1, double y1, double x2, double y2) {
    return std::fabs(x1 - x2) <= EPSILON &&
           std::fabs(y1 - y2) <= EPSILON;
}

// monotonically increases with real angle, but doesn't need expensive trigonometry
inline double pseudo_angle(const double dx, const double dy) {
    const double p = dx / (std::abs(dx) + std::abs(dy));
    return (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0; // [0..1)
}

void Delaunator::triangulate(double* in_coords, const size_t n)
{
    // Initialize
    coords = in_coords;
    num_points = n;
    triangles.clear();
    halfedges.clear();
    hull_prev.clear();
    hull_next.clear();
    hull_tri.clear();
    hull_start = 0;
    m_hash.clear();
    m_center_x = 0;
    m_center_y = 0;
    m_hash_size = 0;
    m_edge_stack.clear();

    if(n == 0) {
        return;
    }

    // Compute

    double max_x = std::numeric_limits<double>::min();
    double max_y = std::numeric_limits<double>::min();
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    std::vector<std::size_t> ids;
    ids.reserve(n);

    for (std::size_t i = 0; i < n; i++) {
        const double x = coords[2 * i];
        const double y = coords[2 * i + 1];

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;

        ids.push_back(i);
    }
    const double cx = (min_x + max_x) / 2;
    const double cy = (min_y + max_y) / 2;
    double min_dist = std::numeric_limits<double>::max();

    std::size_t i0 = INVALID_INDEX;
    std::size_t i1 = INVALID_INDEX;
    std::size_t i2 = INVALID_INDEX;

    // pick a seed point close to the centroid
    for (std::size_t i = 0; i < n; i++) {
        const double d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist) {
            i0 = i;
            min_dist = d;
        }
    }

    const double i0x = coords[2 * i0];
    const double i0y = coords[2 * i0 + 1];

    min_dist = std::numeric_limits<double>::max();

    // find the point closest to the seed
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0) continue;
        const double d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist && d > 0.0) {
            i1 = i;
            min_dist = d;
        }
    }

    double i1x = coords[2 * i1];
    double i1y = coords[2 * i1 + 1];

    double min_radius = std::numeric_limits<double>::max();

    // find the third point which forms the smallest circumcircle with the first two
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0 || i == i1) continue;

        const double r = circumradius(
            i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);

        if (r < min_radius) {
            i2 = i;
            min_radius = r;
        }
    }

    if (!(min_radius < std::numeric_limits<double>::max())) {
        throw std::runtime_error("not triangulation");
    }

    double i2x = coords[2 * i2];
    double i2y = coords[2 * i2 + 1];

    if (orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
        std::swap(i1, i2);
        std::swap(i1x, i2x);
        std::swap(i1y, i2y);
    }

    std::tie(m_center_x, m_center_y) = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y);

    // sort the points by distance from the seed triangle circumcenter
    std::sort(ids.begin(), ids.end(), compare{ coords, m_center_x, m_center_y });

    // initialize a hash table for storing edges of the advancing convex hull
    m_hash_size = static_cast<std::size_t>(std::llround(std::ceil(std::sqrt(n))));
    m_hash.resize(m_hash_size);
    std::fill(m_hash.begin(), m_hash.end(), INVALID_INDEX);

    // initialize arrays for tracking the edges of the advancing convex hull
    hull_prev.resize(n);
    hull_next.resize(n);
    hull_tri.resize(n);

    hull_start = i0;

    size_t hull_size = 3;

    hull_next[i0] = hull_prev[i2] = i1;
    hull_next[i1] = hull_prev[i0] = i2;
    hull_next[i2] = hull_prev[i1] = i0;

    hull_tri[i0] = 0;
    hull_tri[i1] = 1;
    hull_tri[i2] = 2;

    m_hash[hash_key(i0x, i0y)] = i0;
    m_hash[hash_key(i1x, i1y)] = i1;
    m_hash[hash_key(i2x, i2y)] = i2;

    std::size_t max_triangles = n < 3 ? 1 : 2 * n - 5;
    triangles.reserve(max_triangles * 3);
    halfedges.reserve(max_triangles * 3);
    add_triangle(i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
    double xp = std::numeric_limits<double>::quiet_NaN();
    double yp = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t k = 0; k < n; k++) {
        const std::size_t i = ids[k];
        const double x = coords[2 * i];
        const double y = coords[2 * i + 1];

        // skip near-duplicate points
        if (k > 0 && check_pts_equal(x, y, xp, yp)) continue;
        xp = x;
        yp = y;

        // skip seed triangle points
        if (
            check_pts_equal(x, y, i0x, i0y) ||
            check_pts_equal(x, y, i1x, i1y) ||
            check_pts_equal(x, y, i2x, i2y)) continue;

        // find a visible edge on the convex hull using edge hash
        std::size_t start = 0;

        size_t key = hash_key(x, y);
        for (size_t j = 0; j < m_hash_size; j++) {
            start = m_hash[fast_mod(key + j, m_hash_size)];
            if (start != INVALID_INDEX && start != hull_next[start]) break;
        }

        start = hull_prev[start];
        size_t e = start;
        size_t q;

        while (q = hull_next[e], !orient(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1])) { //TODO: does it works in a same way as in JS
            e = q;
            if (e == start) {
                e = INVALID_INDEX;
                break;
            }
        }

        if (e == INVALID_INDEX) continue; // likely a near-duplicate point; skip it

        // add the first triangle from the point
        std::size_t t = add_triangle(
            e,
            i,
            hull_next[e],
            INVALID_INDEX,
            INVALID_INDEX,
            hull_tri[e]);

        hull_tri[i] = legalize(t + 2);
        hull_tri[e] = t;
        hull_size++;

        // walk forward through the hull, adding more triangles and flipping recursively
        std::size_t next = hull_next[e];
        while (
            q = hull_next[next],
            orient(x, y, coords[2 * next], coords[2 * next + 1], coords[2 * q], coords[2 * q + 1])) {
            t = add_triangle(next, i, q, hull_tri[i], INVALID_INDEX, hull_tri[next]);
            hull_tri[i] = legalize(t + 2);
            hull_next[next] = next; // mark as removed
            hull_size--;
            next = q;
        }

        // walk backward from the other side, adding more triangles and flipping
        if (e == start) {
            while (
                q = hull_prev[e],
                orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1])) {
                t = add_triangle(q, i, e, INVALID_INDEX, hull_tri[e], hull_tri[q]);
                legalize(t + 2);
                hull_tri[q] = t;
                hull_next[e] = e; // mark as removed
                hull_size--;
                e = q;
            }
        }

        // update the hull indices
        hull_prev[i] = e;
        hull_start = e;
        hull_prev[next] = i;
        hull_next[e] = i;
        hull_next[i] = next;

        m_hash[hash_key(x, y)] = i;
        m_hash[hash_key(coords[2 * e], coords[2 * e + 1])] = e;
    }
}

double Delaunator::get_hull_area() {
    std::vector<double> hull_area;
    size_t e = hull_start;
    do {
        hull_area.push_back((coords[2 * e] - coords[2 * hull_prev[e]]) * (coords[2 * e + 1] + coords[2 * hull_prev[e] + 1]));
        e = hull_next[e];
    } while (e != hull_start);
    return sum(hull_area);
}

std::size_t Delaunator::legalize(std::size_t a) {
    std::size_t i = 0;
    std::size_t ar = 0;
    m_edge_stack.clear();

    // recursion eliminated with a fixed-size stack
    while (true) {
        const size_t b = halfedges[a];

        /* if the pair of triangles doesn't satisfy the Delaunay condition
        * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
        * then do the same check/flip recursively for the new pair of triangles
        *
        *           pl                    pl
        *          /||\                  /  \
        *       al/ || \bl            al/    \a
        *        /  ||  \              /      \
        *       /  a||b  \    flip    /___ar___\
        *     p0\   ||   /p1   =>   p0\---bl---/p1
        *        \  ||  /              \      /
        *       ar\ || /br             b\    /br
        *          \||/                  \  /
        *           pr                    pr
        */
        const size_t a0 = 3 * (a / 3);
        ar = a0 + (a + 2) % 3;

        if (b == INVALID_INDEX) {
            if (i > 0) {
                i--;
                a = m_edge_stack[i];
                continue;
            } else {
                //i = INVALID_INDEX;
                break;
            }
        }

        const size_t b0 = 3 * (b / 3);
        const size_t al = a0 + (a + 1) % 3;
        const size_t bl = b0 + (b + 2) % 3;

        const std::size_t p0 = triangles[ar];
        const std::size_t pr = triangles[a];
        const std::size_t pl = triangles[al];
        const std::size_t p1 = triangles[bl];

        const bool illegal = in_circle(
            coords[2 * p0],
            coords[2 * p0 + 1],
            coords[2 * pr],
            coords[2 * pr + 1],
            coords[2 * pl],
            coords[2 * pl + 1],
            coords[2 * p1],
            coords[2 * p1 + 1]);

        if (illegal) {
            triangles[a] = p1;
            triangles[b] = p0;

            auto hbl = halfedges[bl];

            // edge swapped on the other side of the hull (rare); fix the halfedge reference
            if (hbl == INVALID_INDEX) {
                std::size_t e = hull_start;
                do {
                    if (hull_tri[e] == bl) {
                        hull_tri[e] = a;
                        break;
                    }
                    e = hull_next[e];
                } while (e != hull_start);
            }
            link(a, hbl);
            link(b, halfedges[ar]);
            link(ar, bl);
            std::size_t br = b0 + (b + 1) % 3;

            if (i < m_edge_stack.size()) {
                m_edge_stack[i] = br;
            } else {
                m_edge_stack.push_back(br);
            }
            i++;

        } else {
            if (i > 0) {
                i--;
                a = m_edge_stack[i];
                continue;
            } else {
                break;
            }
        }
    }
    return ar;
}

inline std::size_t Delaunator::hash_key(const double x, const double y) const {
    const double dx = x - m_center_x;
    const double dy = y - m_center_y;
    return fast_mod(
        static_cast<std::size_t>(std::llround(std::floor(pseudo_angle(dx, dy) * static_cast<double>(m_hash_size)))),
        m_hash_size);
}

std::size_t Delaunator::add_triangle(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t a,
    std::size_t b,
    std::size_t c) {
    std::size_t t = triangles.size();
    triangles.push_back(i0);
    triangles.push_back(i1);
    triangles.push_back(i2);
    link(t, a);
    link(t + 1, b);
    link(t + 2, c);
    return t;
}

void Delaunator::link(const std::size_t a, const std::size_t b) {
    std::size_t s = halfedges.size();
    if (a == s) {
        halfedges.push_back(b);
    } else if (a < s) {
        halfedges[a] = b;
    } else {
        throw std::runtime_error("Cannot link edge");
    }
    if (b != INVALID_INDEX) {
        std::size_t s2 = halfedges.size();
        if (b == s2) {
            halfedges.push_back(a);
        } else if (b < s2) {
            halfedges[b] = a;
        } else {
            throw std::runtime_error("Cannot link edge");
        }
    }
}

} //namespace delaunator
