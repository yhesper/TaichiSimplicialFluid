'''
------
References:
http://www.geometry.caltech.edu/pubs/ETKSD07.pdf
https://github.com/ltt1598/fancy_stable_fluids
https://github.com/haxiomic/GPU-Fluid-Experiments
'''

import taichi as ti
from helper import *
import numpy as np
import random
import math
import meshtaichi_patcher as Patcher
import colorsys
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="meshes/bunny.obj")
parser.add_argument('--arch', default='cuda')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))
vec3f = ti.types.vector(3, ti.f32)
vec4f = ti.types.vector(4, ti.f32)

model = Patcher.load_mesh(args.model, relations=["EV", "VV", "VF", "FV", "EF", "FE", "VE", "FF"])
model.verts.place({'x'   : vec3f, 
                   'vel' : vec3f, 
                   'cn'  : vec3f})
model.faces.place({'vel'      : vec3f,
                   'area'     : ti.f32, 
                   'normal'   : vec3f, 
                   'momentum' : vec3f, 
                   'mean_edge_length' : ti.f32})
model.edges.place({'cotan'    : ti.f32, 
                   'normal'   : vec3f, 
                   'momentum' : vec3f})

model.verts.x.from_numpy(model.get_position_as_numpy())

nv = len(model.verts)
ne = len(model.edges)
nf = len(model.faces)
indices = ti.field(dtype=ti.i32, shape = nf * 3)

@ti.kernel
def get_indices():
    for f in model.faces:
        for j in ti.static(range(3)):
            indices[f.id * 3 + j] = f.verts[j].id
get_indices()


fv = ti.Vector.field(3, ti.i32, shape=nf)

@ti.kernel
def export_fv():
    for f in model.faces:
        fv[f.id] = ti.Vector([f.verts[0].id, f.verts[1].id, f.verts[2].id])
        a = model.verts.x[fv[f.id][0]]
        b = model.verts.x[fv[f.id][1]]
        c = model.verts.x[fv[f.id][2]]

export_fv()


ff = ti.Vector.field(3, ti.i32, shape=nf)

@ti.kernel
def export_ff():
	for f in model.faces:
		ff[f.id] = ti.Vector([f.faces[0].id, f.faces[1].id, f.faces[2].id])
export_ff()


@ti.kernel
def fill_faces_attributes():
    for f in model.faces:
        v0 = f.verts[0]
        v1 = f.verts[1]
        v2 = f.verts[2]
        A = v1.x - v0.x
        B = v2.x - v0.x
        n = ti.math.cross(A, B)
        f.area = ti.math.length(n) * 0.5
        # hardcode a test for making normal vector outwards
        # sadly only works for convex models centered at origin
        if (ti.math.dot(n, v0.x) >= 0.0 and ti.math.dot(n, v1.x) >= 0.0 and ti.math.dot(n, v2.x) >= 0.0):
            f.normal = (n / ti.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]))
        else:
            f.normal = -(n / ti.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]))
        f.mean_edge_length = mean_edge_length(v0.x, v1.x, v2.x)
fill_faces_attributes()

def get_min_edge_length():
    smallest = 100.0
    s = 0.0
    for i in range(nf):
        x = model.faces.mean_edge_length[i]
        s += x
        if (x < smallest):
            smallest = x
    return smallest
min_edge_length = get_min_edge_length()

def fix_normal_direction():
    visited = [False] * nf
    queue = []
    s = 0
    queue.append(s)
    visited[s] = True

    while queue:
        s = queue.pop(0)
        n = model.faces.normal[s]
        nb0, nb1, nb2 = ff[s][0], ff[s][1], ff[s][2]
        n0, n1, n2 = model.faces.normal[nb0], model.faces.normal[nb1], model.faces.normal[nb2]
        if visited[nb0] == False:
            queue.append(nb0)
            visited[nb0] = True
            if (n[0]*n0[0] + n[1]*n0[1]+n[2]*n0[2] < 0.0):
                model.faces.normal[nb0] = -n0
        if visited[nb1] == False:
            queue.append(nb1)
            visited[nb1] = True
            if (n[0]*n1[0] + n[1]*n1[1]+n[2]*n1[2] < 0.0):
                model.faces.normal[nb1] = -n1
        if visited[nb2] == False:
            queue.append(nb2)
            visited[nb2] = True  
            if (n[0]*n2[0] + n[1]*n2[1]+n[2]*n2[2] < 0.0):
                model.faces.normal[nb2] = -n2
    
fix_normal_direction()

@ti.kernel
def fill_edges_attributes():
    for e in model.edges:
        v0 = e.verts[0]
        v1 = e.verts[1]
        f0 = e.faces[0]
        f1 = e.faces[1]
        v2 = f0.verts[ 2 if (v0.id == f0.verts[0].id and v1.id == f0.verts[1].id) or (v1.id == f0.verts[0].id and v0.id ==f0.verts[1].id) 
                else (
                    1 if (v0.id == f0.verts[0].id and v1.id == f0.verts[2].id) or (v1.id == f0.verts[0].id and v0.id ==f0.verts[2].id) 
                    else 0
                )]
        v3 = f1.verts[ 2 if (v0.id == f1.verts[0].id and v1.id == f1.verts[1].id) or (v1.id == f1.verts[0].id and v0.id ==f1.verts[1].id) 
                else (
                    1 if (v0.id == f1.verts[0].id and v1.id == f1.verts[2].id) or (v1.id == f1.verts[0].id and v0.id ==f1.verts[2].id) 
                    else 0
                )]
        A0 = v0.x - v2.x
        B0 = v1.x - v2.x
        A1 = v0.x - v3.x
        B1 = v1.x - v3.x
        val = ti.math.dot(A0, B0)/ti.math.length(ti.math.cross(A0, B0)) + ti.math.dot(A1, B1)/ti.math.length(ti.math.cross(A1, B1))
        e.cotan = val
        e.normal = (f0.normal + f1.normal) * 0.5

fill_edges_attributes()

# compute curvature normals for vertices
@ti.kernel
def fill_vert_normals():
    for vert in model.verts:
        cn = vec3f(0.0)
        for i in range(vert.edges.size):
            e = vert.edges[i]
            v1 = e.verts[ 1 if e.verts[0].id == vert.id else 0]
            cn += e.cotan * (vert.x - v1.x)
        vert.cn = ti.math.normalize(cn)
fill_vert_normals()

'''
##########################################
fluid
##########################################
'''

flux = ti.field(ti.f32, shape=ne)  
circulation = ti.field(ti.f32, shape=ne)

vorticity =  ti.field(ti.f32, shape=nv)  
new_vorticity = ti.field(ti.f32, shape=nv)

@ti.kernel
def advect(dt : ti.f32):
    for vert in model.verts:
        vel = vec3f(0.0)
        w = 0.0
        for i in range(vert.faces.size):
            f = vert.faces[i]
            u = f.vel
            u = u - ti.math.dot(u, vert.cn) * vert.cn
            vel += f.area * u
            w += f.area
        vel /= w
        vert.vel = vel
        p = vec3f(0.0,0.0,0.0) - dt * vel
        # find which face p is in
        u = 0.0
        v = 0.0
        w = 0.0
        x = 0.0
        y = 0.0
        z = 0.0
        find = False
        # rely on the fact that this loop is not parallel
        for i in range(vert.faces.size):
            f = vert.faces[i]
            a = f.verts[0].x
            b = f.verts[1].x
            c = f.verts[2].x
            a = a - vert.x
            b = b - vert.x
            c = c - vert.x
            a = a - ti.math.dot(a, vert.cn) * vert.cn
            b = b - ti.math.dot(b, vert.cn) * vert.cn
            c = c - ti.math.dot(c, vert.cn) * vert.cn
            if (ti.math.length(p-a)<1e-8):
                u = 1.0
                v = 0.0
                w = 0.0
            if (ti.math.length(p-b)<1e-8):
                u = 0.0
                v = 1.0
                w = 0.0
            if (ti.math.length(p-c)<1e-8):
                u = 0.0
                v = 0.0
                w = 1.0
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
            
            if u >= 0.0 and v>=0.0 and w>=0.0:
                find = True
                x = vorticity[f.verts[0].id]
                y = vorticity[f.verts[1].id]
                z = vorticity[f.verts[2].id]
                new_vorticity[vert.id]= u * x + v*y + w*z
        if (not find):
            new_vorticity[vert.id] = vorticity[vert.id]
    
    # # end computations, replace old vorticity
    for i in range(nv):
        vorticity[i] = new_vorticity[i]

'''
whiteny interpolation to extract velocity from flux
note that we need to carefuly assign edge orientations
we assuming each edge is pointed from e.verts[0] to e.verts[1]

'''
@ti.kernel
def flux2vel(dt: ti.f32) -> ti.i8:
    exceed_cfl = 0
    for f in model.faces:
        N = f.normal
        A = f.area
        e0 = f.edges[0]
        e1 = f.edges[1]
        e2 = f.edges[2]
        s0, s1, s2 = 1.0, 1.0, 1.0
        ei = e0.id
        ej = e1.id
        ek = e2.id # dummy value
        pi = e0.verts[0].id
        pj = e0.verts[1].id
        pk = e1.verts[0].id
        eij = model.verts.x[pj] - model.verts.x[pi] # same direction as e0
        ejk = model.verts.x[pj] - model.verts.x[pi] # dummy value
        eki = model.verts.x[pj] - model.verts.x[pi] # dummy value
        if (e1.verts[0].id == pj):
            pk = e1.verts[1].id
            if (e2.verts[0].id == pi):
                s2 = -1.0
        elif (e1.verts[1].id == pj):
            pk = e1.verts[0].id
            s1 = -1.0
            if (e2.verts[0].id == pi):
                s2 = -1.0
        elif (e2.verts[0].id == pj):
            pk = e2.verts[1].id
            ej, ek = e2.id, e1.id
            if (e1.verts[0].id == pi):
                s2 = -1.0
        else:
            # e2.verts[1] == pj
            pk = e2.verts[0].id
            s1 = -1.0
            ej, ek = e2.id, e1.id
            if (e1.verts[0].id == pi):
                s2 = -1.0
        # check orientation
        ejk = model.verts.x[pk] - model.verts.x[pj]
        eki = model.verts.x[pi] - model.verts.x[pk]

        if (ti.math.dot(ti.math.cross(eij, ejk), N) < 0):
            # change the entire orientation
            s0 = -s0
            s1 = -s1
            s2 = -s2
        wi = flux[ei] * s0
        wj = flux[ej] * s1
        wk = flux[ek] * s2
        s = (1.0 / (6.0 * A)) * ti.math.cross(N, wi*(eki-ejk) + wj*(eij-eki) + wk*(ejk-eij))

        vel = s
        vel = ti.math.cross(N, vel)
        if (ti.math.length(vel * dt) > f.mean_edge_length):
            exceed_cfl = 1
        f.vel = vel
    return exceed_cfl
        
@ti.kernel
def decay_fluid():
    for f in model.faces:
        f.vel = 0.75 * f.vel
    for i in range(nv):
        vorticity[i] = 0.75 * vorticity[i]

@ti.kernel
def negate_velocity():
    for f in model.faces:
        f.vel = -1.0 * f.vel

@ti.kernel
def print_velocity():
    print("new iteration")
    for f in model.faces:
        print("velocity", f.vel)



phi_n = ti.field(dtype=ti.f32, shape=nv)
phi_bar = ti.field(dtype=ti.f32, shape=nv)
phi_mid = ti.field(dtype=ti.f32, shape=nv)
rho_bar = ti.field(dtype=ti.f32, shape=nv)


@ti.kernel
def add(ans: ti.template(), l:ti.f32, a: ti.template(), k: ti.f32, b: ti.template()):
    for i in range(nv):
        ans[i] = l*a[i] + k*b[i]

pk = ti.field(dtype=ti.f32, shape=nv)
Lxk = ti.field(dtype=ti.f32, shape=nv)
Lpk = ti.field(dtype=ti.f32, shape=nv)

xk = ti.field(dtype=ti.f32, shape=nv)
rk = ti.field(dtype=ti.f32, shape=nv)
p = ti.field(dtype=ti.f32, shape=nv)
z = ti.field(dtype=ti.f32, shape=nv)
w = ti.field(dtype=ti.f32, shape=nv)
Ax = ti.field(dtype=ti.f32, shape=nv)
Ap = ti.field(dtype=ti.f32, shape=nv)


@ti.kernel
def compute_rk_norm() -> ti.f32:
    ans = 0.0
    for i in range(nv):
        ans += rk[i] * rk[i]
    return ti.sqrt(ans)

@ti.kernel
def compute_b_norm() -> ti.f32:
    ans = 0.0
    for i in range(nv):
        ans += rho_bar[i] * rho_bar[i]
    return ti.sqrt(ans)

@ti.kernel
def L_mul(ans : ti.template(), x : ti.template()):
    for v0 in model.verts:
        w = 0.0
        val = 0.0
        for i in range(v0.edges.size):
            e = v0.edges[i]
            v1 = e.verts[1 if e.verts[0].id == v0.id else 0]
            w += e.cotan
            val +=  -e.cotan * x[v1.id]
        ans[v0.id] = val + w * x[v0.id]


@ti.kernel
def dot_field(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    for i in range(nv):
        ans += a[i] * b[i]
    return ans

@ti.kernel
def negate_field(a: ti.template(), b: ti.template()):
    for i in range(nv):
        a[i] = -b[i]

@ti.kernel
def Minv_mul(ans : ti.template(), x : ti.template()):
    for v0 in model.verts:
        w = 0.0
        for i in range(v0.edges.size):
            e = v0.edges[i]
            w +=  e.cotan
        ans[v0.id] = (1.0 / w) * x[v0.id]


@ti.kernel
def scale_xk():
    for i in range(nv):
        xk[i] = xk[i] * 0.8


# no preconditioning
def cg():
    L_mul(Lxk, xk)
    add(rk, 1.0, Lxk, -1.0, rho_bar)
    negate_field(pk, rk)
    # print(xk, rho_bar, Lxk, rk, pk)
    rk_norm = compute_rk_norm()
    n_iter = 20
    epsilon = 1e-5
    if rk_norm <=  epsilon:
        return
    b_norm = compute_b_norm()
    for i in range(n_iter):
        L_mul(Lpk, pk)
        rk2 = dot_field(rk, rk)
        alpha = rk2 / dot_field(pk, Lpk)
        add(xk, 1.0, xk, alpha, pk)
        add(rk, 1.0, rk, alpha, Lpk)
        beta = dot_field(rk, rk) / rk2
        add(pk, -1.0, rk, beta, pk)
        rk_norm = compute_rk_norm()
        if rk_norm <=  epsilon * b_norm:
            break


bfecc_const = 0.3

@ti.kernel
def set_phi_mid():
    for i in range(nv):
        phi_mid[i] = phi_n[i] + bfecc_const * (phi_n[i] - phi_bar[i])



@ti.kernel
def mean_vorticity() -> ti.f32:
    ans = 0.0
    for i in range(nv):
        ans += vorticity[i]
    return ans

@ti.kernel
def set_flux():
    ''' flux is d0 * phi(xk)'''
    for e in model.edges:
        v0 = e.verts[0]
        v1 = e.verts[1]
        flux[e.id] = xk[v1.id] - xk[v0.id]
        
@ti.kernel
def set_rho_bar():
    m = 0.0
    for i in range(nv):
        m += vorticity[i]
    m = m / nv
    for i in range(nv):
        rho_bar[i] = vorticity[i] - m


'''
##########################################
dye
##########################################
'''

dye = ti.Vector.field(n=3, dtype=ti.f32, shape=nv)
dye_base = ti.Vector.field(n=3, dtype=ti.f32, shape=nv)
new_dye = ti.Vector.field(n=3, dtype=ti.f32, shape=nv)
@ti.kernel
def init_dye():
    for i in range (nv):
        dye_base[i] = vec3f(0.1, 0.1, 0.1)
init_dye()

vertex_color = ti.Vector.field(n=3, dtype=ti.f32, shape=nv)
@ti.kernel
def set_vertex_color():
    for i in range (nv):
        vertex_color[i] = dye_base[i] + ti.max(0.0, dye[i])

# advect_dye code is the same as advect
@ti.kernel
def advect_dye(dt : ti.f32, dissipation : ti.f32):
    decay = 1.0 + dissipation * dt
    for vert in model.verts:
        vel = vec3f(0.0)
        w = 0.0
        for i in range(vert.faces.size):
            f= vert.faces[i]
            u = f.vel
            u = u - ti.math.dot(u, vert.cn) * vert.cn
            vel += f.area * u
            w += f.area
        vel /= w
        p = vec3f(0.0,0.0,0.0) - dt * vel
        # find which face p is in
        u = 0.0
        v = 0.0
        w = 0.0
        find = False
        for i in range(vert.faces.size):
            f= vert.faces[i]
            a = f.verts[0].x
            b = f.verts[1].x
            c = f.verts[2].x
            a = a - vert.x
            b = b - vert.x
            c = c - vert.x
            a = a - ti.math.dot(a, vert.cn) * vert.cn
            b = b - ti.math.dot(b, vert.cn) * vert.cn
            c = c - ti.math.dot(c, vert.cn) * vert.cn
            if (ti.math.length(p-a)<1e-8):
                u = 1.0
                v = 0.0
                w = 0.0
            if (ti.math.length(p-b)<1e-8):
                u = 0.0
                v = 1.0
                w = 0.0
            if (ti.math.length(p-c)<1e-8):
                u = 0.0
                v = 0.0
                w = 1.0
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
            
            if u >= 0.0 and v>=0.0 and w>=0.0:
                find = True
                x = dye[f.verts[0].id]
                y = dye[f.verts[1].id]
                z = dye[f.verts[2].id]
                new_dye[vert.id]= u * x + v*y + w*z
        if (not find):
            new_dye[vert.id] = clamp(dye[vert.id], 0.0, 1.0)
            
    for i in range(nv):
        dye[i] = ti.min(ti.max(new_dye[i], 0.0), 1.0)

def generate_color():
    c = vec3f(colorsys.hsv_to_rgb(np.random.random(), 1.0, 1.0))
    c *= 1.0
    return c




'''
##########################################
particles
##########################################
'''
pnum = 10000
particle_field = ti.Struct.field(
        {'pos':vec3f, 'bc':vec3f, 'fid':ti.i32}, shape = (pnum,))

particle_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=pnum)

@ti.kernel
def random_particle(i : ti.i32, fid : ti.i32, u : ti.f32, v : ti.f32):
    particle_field[i].fid = fid
    particle_field[i].bc = vec3f(u, v, 1.0-u-v)
    a = model.verts.x[fv[fid][0]]
    b = model.verts.x[fv[fid][1]]
    c = model.verts.x[fv[fid][2]]
    particle_field[i].pos = u * a + v * b + c * (1.0-u-v)

for i in range(pnum):
    fid = random.randint(0,nf)
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    while (u+v > 1.0): 
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
    random_particle(i, fid, u, v)

@ti.func
def has_edge(p : ti.i32, q : ti.i32, fid : ti.i32) -> bool:
    count = 0
    if (p == fv[fid][0]):
        count += 1
    if (p == fv[fid][1]):
        count += 1    
    if (p == fv[fid][2]):
        count += 1
    if (q == fv[fid][0]):
        count += 1
    if (q == fv[fid][1]):
        count += 1    
    if (q == fv[fid][2]):
        count += 1
    return count == 2



@ti.kernel
def move_particles(dt0 : ti.f32):
    dt = dt0
    for i in range(pnum):
        par = particle_field[i]
        vel = model.faces.vel[par.fid]
        pos = par.pos + vel * dt
        pi = model.verts.x[fv[par.fid][0]]
        pj = model.verts.x[fv[par.fid][1]]
        pk = model.verts.x[fv[par.fid][2]]
        bc = barycentric(pos, pi, pj, pk)
        u = bc[0]
        v = bc[1]
        w = bc[2]
        f_ij, f_jk, f_ki = ff[par.fid][0], ff[par.fid][1], ff[par.fid][2] #dummy values
        if (has_edge(fv[par.fid][0], fv[par.fid][1], ff[par.fid][1])):
            f_ij = ff[par.fid][1]
        elif (has_edge(fv[par.fid][0], fv[par.fid][1], ff[par.fid][2])):
            f_ij = ff[par.fid][2]
        if (has_edge(fv[par.fid][1], fv[par.fid][2], ff[par.fid][0])):
            f_jk = ff[par.fid][0]
        elif (has_edge(fv[par.fid][1], fv[par.fid][2], ff[par.fid][2])):
            f_jk = ff[par.fid][2]
        if (has_edge(fv[par.fid][2], fv[par.fid][0], ff[par.fid][0])):
            f_ki = ff[par.fid][0]
        elif (has_edge(fv[par.fid][2], fv[par.fid][0], ff[par.fid][1])):
            f_ki = ff[par.fid][1]

        if 0.0 <= u and u <= 1.0 and 0.0 <= v and v <= 1.0 and 0.0 <= w and w <= 1.0:
            # we are in the original triangle
            par.pos = pos
            par.bc = bc
        else: #we are going to a new traingle
            if u < 0:
                # p is out of edge pjpk
                r = pk - pj
                s = pos - pi
                x = ti.math.length(ti.math.cross((pi - pj), s)) / ti.math.length(ti.math.cross(r, s))
                crossing = pj + x * r
                ddt = (crossing - par.pos)[0] / vel[0]
                if (ddt < 0):
                    ddt = 0
                new_fid = f_jk
                new_pos = crossing + (dt-ddt) *  model.faces.vel[new_fid]
                bc = barycentric(new_pos, model.verts.x[fv[new_fid][0]], model.verts.x[fv[new_fid][1]], model.verts.x[fv[new_fid][2]])
                par.pos = new_pos
                par.fid = new_fid
                par.bc = bc
                pos = crossing
                dt = dt - ddt
            elif v < 0:
                # p is out of edge pipk
                r = pk - pi
                s = pos - pj
                x = ti.math.length(ti.math.cross((pj - pi), s)) / ti.math.length(ti.math.cross(r, s))
                crossing = pi + x * r
                ddt = (crossing - par.pos)[0] / vel[0]
                if (ddt < 0):
                    ddt = 0
                new_fid = f_ki
                new_pos = crossing + (dt-ddt) *  model.faces.vel[new_fid]
                bc = barycentric(new_pos, model.verts.x[fv[new_fid][0]], model.verts.x[fv[new_fid][1]], model.verts.x[fv[new_fid][2]])
                par.pos = new_pos
                par.fid = new_fid
                par.bc = bc
                pos = crossing
                dt = dt - ddt
            else:
                #w < 0
                # p is out of edge pipj
                r = pj - pi
                s = pos - pk
                x = ti.math.length(ti.math.cross((pk - pi), s)) / ti.math.length(ti.math.cross(r, s))
                crossing = pi + x * r
                ddt = (crossing - par.pos)[0] / vel[0]
                if ddt < 0:
                    ddt = 0
                new_fid = f_ij
                new_pos = crossing + (dt-ddt) *  model.faces.vel[new_fid]
                bc = barycentric(new_pos, model.verts.x[fv[new_fid][0]], model.verts.x[fv[new_fid][1]], model.verts.x[fv[new_fid][2]])
                par.pos = new_pos
                par.fid = new_fid
                par.bc = bc
                pos = crossing
                dt = dt - ddt
        u = par.bc[0]
        v = par.bc[1]
        w = par.bc[2]
        new_p = par.pos
        A = model.verts.x[fv[par.fid][0]]
        B = model.verts.x[fv[par.fid][1]]
        C = model.verts.x[fv[par.fid][2]]
        if (u < 0):
            t = ti.math.dot(new_p-B, C - B)/ti.math.dot(C-B, C - B)
            if (t<0):
                t = 0
            if (t>1):
                t = 1
            u, v, w = 0, 1-t, t
            par.bc = vec3f(u, v, w)
            par.pos = u * A + v * B + w * C
        elif v < 0:
            t = ti.math.dot(new_p-C, A - C)/ti.math.dot(A-C, A - C)
            if (t<0):
                t = 0
            if (t>1):
                t = 1
            u, v, w = t, 0, 1-t
            par.bc = vec3f(u, v, w)
            par.pos = u * A + v * B + w * C
        elif w < 0:
            t = ti.math.dot(new_p-A,  B - A)/ti.math.dot(B-A,B - A)
            if (t<0):
                t = 0
            if (t>1):
                t = 1
            u, v, w = 1-t, t, 0
            par.bc = vec3f(u, v, w)
            par.pos = u * A + v * B + w * C
        particle_field[i] = par
        '''set color based on speed'''
        vel = model.faces.vel[par.fid]
        speed = ti.math.length(vel)
        x = clamp(speed/20, 0., 1.)
        c = mix(vec3f(40.4, 0.0, 35.0) / 300.0, vec3f(0.2, 47.8, 100) / 100.0, x)  + (vec3f(63.1, 92.5, 100) / 100.) * x*x*x * 0.1
        if (c[0] > 0.1 or c[1] > 0.1 or c[2] > 0.1):
            particle_colors[i] = c
        else:
            particle_colors[i] = vec3f(0.1, 0.1, 0.1)

@ti.kernel
def set_particle_colors():
    for i in range(pnum):
        par = particle_field[i]
        vel = model.faces.vel[par.fid]
        speed = ti.math.length(vel)
        x = clamp(speed/20, 0., 1.)
        # print(x)
        c = mix(vec3f(40.4, 0.0, 35.0) / 300.0, vec3f(0.2, 47.8, 100) / 100.0, x)  + (vec3f(63.1, 92.5, 100) / 100.) * x*x*x * 0.1
        if (c[0] > 0.1 or c[1] > 0.1 or c[2] > 0.1):
            particle_colors[i] = c
        else:
            particle_colors[i] = vec3f(0.1, 0.1, 0.1)



'''
##########################################
mouse
##########################################
'''
@ti.kernel
def ray_from_mouse(mouse_x : ti.f32, mouse_y : ti.f32, cam2world:ti.types.matrix(4,4,ti.f32)) -> vec3f:
    p = ti.Matrix([[(mouse_x-0.5)], [(mouse_y-0.5)], [-1.0], [0.0]])
    ray = to_hetero(cam2world @ p)    
    ray = ti.math.normalize(ray)
    return ray


x = ti.field(ti.f32)
block = ti.root.pointer(ti.ij, (nf,1))
pixel = block.dense(ti.ij, (1,1))
pixel.place(x)

hit_info = ti.Struct.field({
    "tri": ti.i32,"time": ti.f32,}, shape=(1,))
hit_info[0].time = 1000.0
hit_info[0].tri = -1

'''
to do: add bvh
'''
@ti.kernel
def hit_mesh(ray : vec3f, campos:vec3f):
    for fa in model.faces:
        v0 = fa.verts[0].x
        v1 = fa.verts[1].x
        v2 = fa.verts[2].x
        n = fa.normal
        hit = True
        edge2 = v2 - v0
        edge1 = v1 - v0
        h = ti.math.cross(ray, edge2)
        a = ti.math.dot(edge1, h)
        if (a < 1e-4 and a>-1e-4):
            hit = False
        f = 1.0 / a
        s = campos - v0
        u = f * ti.math.dot(s, h)
        if (u < 0.0 or u > 1.0):
            hit = False
        q = ti.math.cross(s, edge1)
        v = f * ti.math.dot(ray, q)
        if(v < 0.0 or u + v > 1.0) :
            hit = False
        dist = f * ti.math.dot(edge2, q)
        if (hit):
            x[fa.id, 0] = dist

    hit_time = 1000.0
    hit_tri = -1
    for i,j in x:
        hit_time += 0.0
        ti.atomic_min(hit_time, x[i,j])
    for i,j in x:
        if (x[i,j] == hit_time):
            hit_tri = i
    hit_info[0].time = hit_time
    hit_info[0].tri = hit_tri


class MouseData(object):
    def __init__(self):
        self.move = False
        self.mxy = None #nparray
        self.mdelta = None #nparray
        self.color = None #nparray
        self.tri = -1
        self.mxyz = None #nparray
        self.mprev = None

# mouse events
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None 
        self.prev_color = None
        self.mouse_ticks = 0
        self.change_color = False
        self.prev_hit = None #vec3f
        self.prev_tri = -1


    def __call__(self, window, cam2world, campos):
        mouse_data = MouseData()
        if window.is_pressed(ti.ui.LMB):
            mxy = np.array(window.get_cursor_pos(), dtype=np.float32)
            r = ray_from_mouse(float(mxy[0]), float(mxy[1]), cam2world)
            hit_mesh(r, campos)
            tri = hit_info[0].tri
            t = hit_info[0].time
            m_in_world = campos + r * t

            if (tri == -1): # if no hit then it's useless
                mouse_data.move = False
                self.prev_mouse = None
                self.prev_color = None
                return mouse_data

            # change dye color every 10 mouse events
            if self.change_color and self.mouse_ticks > 9:
                self.mouse_ticks = 0
                self.prev_color = generate_color()

            if self.prev_mouse is None: # mouse pressed
                self.mouse_ticks = 0
                self.prev_mouse = mxy
                self.prev_color = generate_color()
                self.prev_hit = m_in_world
                self.prev_tri = tri
            else: # mouse moving
                self.mouse_ticks += 1
                delta_mxy = m_in_world - self.prev_hit
                if norm(delta_mxy) > 1e-4:
                    mouse_data.move = True
                    mouse_data.mxy = mxy
                    mouse_data.mdelta = delta_mxy
                    mouse_data.color = self.prev_color
                    mouse_data.tri = tri
                    mouse_data.mxyz = m_in_world
                    mouse_data.mprev = self.prev_hit
                self.prev_mouse = mxy
                self.prev_hit = m_in_world

        else:
            mouse_data.move = False
            self.prev_mouse = None
            self.prev_color = None
        
        return mouse_data



force_radius = min_edge_length * 4
inv_force_radius = 1.0 / force_radius
dye_radius = 1 / 2000 #old: 0.1/200
inv_dye_radius = 1.0 / dye_radius
aspect_ratio = 1
f_strength = min_edge_length * 1000 * 1.5

@ti.kernel
def splat_velocity1(mpos: vec3f, mdir: vec3f):
    for f in model.faces:
        center = (f.verts[0].x + f.verts[1].x + f.verts[2].x) / 3.0
        dx = center[0] - mpos[0]
        dy = center[1] - mpos[1]
        dz = center[2] - mpos[2]
        d2 = dx * dx + dy * dy + dz * dz
        # project mdir onto f's plane
        mdir_p = mdir - ti.math.dot(mdir, f.normal)
        mdir_p = ti.math.normalize(mdir_p)
        multiplier = ti.exp(-d2 * inv_force_radius) * f_strength * f.mean_edge_length
        f.momentum = multiplier * mdir_p

@ti.kernel
def splat_velocity2():
    for e in model.edges:
        f0 = e.faces[0]
        f1 = e.faces[1]
        e.momentum = (f0.momentum + f1.momentum) * 0.5

@ti.kernel
def splat_velocity3():
    for v0 in model.verts:
        x = 0.0
        for i in range(v0.edges.size):
            e = v0.edges[i]
            v1 = e.verts[1 if e.verts[0].id == v0.id else 0]
            dualmomentum = ti.math.cross(e.normal, e.momentum)
            x += e.cotan * ti.math.dot(dualmomentum, v1.x - v0.x)
        vorticity[v0.id] += x

@ti.func
def clamp(x, lo, hi):
    return ti.max(lo, ti.min(x, hi))

@ti.func
def mix(x, y, a):
    return x * (1-a) + y * a



@ti.kernel
def splat_dye(mpos:vec3f, mprev:vec3f, mdir: vec3f, color0:vec3f):
    for v in model.verts:
        
        dx = v.x[0] - mpos[0]
        dy = v.x[1] - mpos[1]
        dz = v.x[2] - mpos[2]
        d2 = dx * dx + dy * dy + dz * dz
        color = ti.exp(-d2 * inv_dye_radius) * color0
        dye[v.id] += color

@ti.kernel
def check_exceed_cfl(dt:ti.f32) -> ti.i8:
    exceed_cfl = 0
    for f in model.faces:
        if (ti.math.length(f.vel * dt) > f.mean_edge_length):
            exceed_cfl = 1
    return exceed_cfl

def apply_impulse(mouse_data, cfl_ok):
    if (mouse_data.tri != -1):
        #  only apply impulse when we hit something
        if (cfl_ok):
            splat_velocity1(mouse_data.mxyz, mouse_data.mdelta)
            splat_velocity2()
            splat_velocity3()
        splat_dye(mouse_data.mxyz, mouse_data.mprev, mouse_data.mdelta, mouse_data.color)


@ti.kernel
def reset_velocities():
    for f in model.faces:
        f.vel = vec3f(0.0)

    

'''
##########################################
set camera
##########################################
'''

win_x = 1000
win_y = 1000

window = ti.ui.Window("simp fluid", (win_x, win_y))
canvas = window.get_canvas()
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(2, 2, 2)
camera.lookat(0, 0, 0)
camera.up(0, 1, 0)
camera.fov(45)


'''
##########################################
run simulation
##########################################
'''

dt = 0.0001


md_gen = MouseDataGen()
frame = 0
exceed_cfl = 0

ti.sync() # hack the ggui bug
while True:
    frame += 1
    ti.deactivate_all_snodes()
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    paused = False
    write_image = False
    show_particles = True
    if window.get_event(ti.ui.PRESS):
        e = window.event
        if e.key == ti.ui.ESCAPE:
            break
        elif e.key == 'r':
            paused = False
            reset_velocities()
            vorticity.fill(0.0)
            flux.fill(0.0)
            dye.fill(0.0)
            camera.position(2, 2, 2)
            camera.lookat(0, 0, 0)
            camera.up(0, 1, 0)
            camera.fov(45)
        elif e.key == 'p':
            paused = True
        elif e.key == "x":
            write_image = True
        elif e.key == 'v':
            show_particles = not show_particles
        elif e.key == 'c':
            md_gen.change_color = not md_gen.change_color


    if not paused:
        view = camera.get_view_matrix()
        view_inv = np.linalg.inv(view)
        campos = vec3f(view_inv[3][0], view_inv[3][1], view_inv[3][2])
        mouse_data = md_gen(window, to_mat4(view_inv), campos)
        '''advect vorticity with bfecc'''
        phi_n.copy_from(vorticity)
        advect(dt/2)
        negate_velocity()
        advect(dt/2)
        phi_bar.copy_from(vorticity)
        set_phi_mid()
        negate_velocity()
        vorticity.copy_from(phi_mid)
        advect(dt/2)

        if mouse_data.move:
            apply_impulse(mouse_data, exceed_cfl == 0)

        '''solve for flux'''
        set_rho_bar()
        cg()
        set_flux()
        exceed_cfl = flux2vel(dt)
        while (exceed_cfl):
            print("Warning: Does not meet Courant–Friedrichs–Lewy condition!")
            decay_fluid()
            exceed_cfl = check_exceed_cfl(dt)

        '''advect dye'''
        advect_dye(dt, 0.9)
        

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.point_light(pos=(0.5, 1.5, -1.5), color=(1, 1, 1))
    scene.point_light(pos=(0.5, 1.5, -0.5), color=(1, 1, 1))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    set_vertex_color()
    scene.mesh(model.verts.x, indices, per_vertex_color = vertex_color)
    
    if (show_particles):
        move_particles(dt)
        scene.particles(particle_field.pos, per_vertex_color = particle_colors, radius = 0.002)
    canvas.scene(scene)
    if (write_image):
        out_file = "screenshots/" + obj_name[7:(len(obj_name)-4)] + ("%d.png" % frame)
        window.write_image(out_file)
        write_image = False

    window.show()
