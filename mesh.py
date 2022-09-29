import taichi as ti
ti.init(arch=ti.cpu)

dt = 5e-3
radius = 1.0
spacing = radius + 0.2
M = 64 # cols of particle rows
N = 64 # rows of particles
particles = ti.Vector.field(3, float, (M * N,))
colors = ti.Vector.field(3, float, (M * N,))
radians = 0

n_triangles = (N - 1) * (M - 1) * 2
# if int(n_triangles) != n_triangles:
#     raise Exception(f'Non-integer triangle count {n_triangles}')
indices = ti.field(int, int(n_triangles) * 3)

def init():
    for m, n in ti.ndrange(M, N):
        particles[m * M + n] = ti.Vector([(m - (M / 2)) * spacing, 0, (n - (N / 2)) * spacing])
        colors[m * M + n] = ti.Vector([m / M, 0, n / N])
        if m < M - 1 and n < N - 1:
            square_id = (m * (M - 1)) + n
            # 1st triangle of the square
            indices[square_id * 6 + 0] = m * M + n
            indices[square_id * 6 + 1] = (m + 1) * M + n
            indices[square_id * 6 + 2] = m * M + (n + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (m + 1) * M + n + 1
            indices[square_id * 6 + 4] = m * M + (n + 1)
            indices[square_id * 6 + 5] = (m + 1) * M + n

@ti.kernel
def run(radians: float):
    for m, n in ti.ndrange(M, N):
        particles[m * M + n].y = ti.cos(radians * dt + 10 * (m/M + n/N)) - 0.5

window = ti.ui.Window('Particles', res=(1280, 720), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, 5, -(M / 2))
camera.lookat(0, 0, 0)
fov = 45
show_wireframe = False

init()

while window.running:
    if window.is_pressed(ti.ui.ESCAPE):
        break
    if window.is_pressed(ti.ui.UP):
        fov = min(150, fov + 0.5)
    if window.is_pressed(ti.ui.DOWN):
        fov = max(1, fov - 0.5)
    if window.is_pressed('z'):
        show_wireframe = not show_wireframe
    if window.is_pressed('x'):
        camera.position(0, 5, -(M / 2))
        camera.lookat(0, 0, 0)
    if window.is_pressed('c'):
        radians = 0
    camera.fov(fov)
    camera.track_user_inputs(window, hold_key=ti.ui.LMB)

    radians += 1
    run(radians)

    # scene.point_light(pos=(0, 5, 0), color=(1, 1, 1))
    scene.ambient_light((1, 1, 1))
    # scene.particles(particles, radius, per_vertex_color=colors)
    scene.mesh(particles, per_vertex_color=colors,
                          indices=indices,
                          show_wireframe=show_wireframe)
    scene.set_camera(camera)
    canvas.scene(scene)
    window.show()