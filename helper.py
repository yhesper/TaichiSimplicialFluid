import taichi as ti
from math import *
import numpy as np
vec3f = ti.types.vector(3, ti.f32)


def norm(v):
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


def cross(u, v):
    return vec3f(u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0])


@ti.func
def mean_edge_length(x, y, z):
    a = x - y
    b = y - z
    c = z - x
    return (ti.math.length(a) + ti.math.length(b) + ti.math.length(c)) / 3.0


@ti.func
def mean(vec):
    n = vec.shape[0]
    s = 0
    for i in range(n):
        s += vec[i]
    return s / n


@ti.func
def to_hetero(m):
    # return vec3f(m[0]/m[3], m[1]/m[3], m[2]/m[3])
    return vec3f(m.xyz)



@ti.func
def barycentric(p, a, b, c) -> vec3f:
    u, v, w = 0.0, 0.0, 0.0
    if (ti.math.distance(p,a)<1e-8):
        u, v, w = 1.0, 0.0, 0.0
    elif (ti.math.distance(p,b)<1e-8):
        u, v, w = 0.0, 1.0, 0.0
    elif (ti.math.distance(p,c)<1e-8):
        u, v, w = 0.0, 0.0, 1.0
    else:
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = ti.math.dot(v0, v0)
        d01 = ti.math.dot(v0, v1)
        d11 = ti.math.dot(v1, v1)
        d20 = ti.math.dot(v2, v0)
        d21 = ti.math.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
    return vec3f(u, v, w)

@ti.func
def compute_curvature_normal(v0):
    n = vec3f(0.0)
    for e in v0.edges:
        v1 = e.verts[ 1 if e.verts[0].id == v0.id else 0]
        n += e.cotan * (v0.pos - v1.pos)
    n = (n / ti.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]))
    return n



def to_mat4(arr):
    M = ti.Matrix([[0] * 4 for _ in range(4)], ti.f32)
    for i in range(4):
        for j in range(4):
            M[i, j] = float(arr[j][i])
    return M