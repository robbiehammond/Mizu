import numpy as np
from scipy.special import gamma
import taichi as ti
import math 

ti.init(arch=ti.cpu)

def W( x, y, z, h ):
	"""
	Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	M, N = x.shape
	r = ti.field(float, shape=(M,N))
	for i in range(M):
		for j in range(N):
			r[i,j] = math.sqrt(x[i,j]**2 + y[i,j]**2 + z[i,j]**2)

	const = (1.0 / (h*math.sqrt(math.pi)))**3 
	for i in range(M):
		for j in range(N):
			r[i, j] = const * pow(math.e, -r[i,j]**2 / h**2)
	
	return r
	
@ti.kernel
def populateR(M: int, N: int, r: ti.template(), x: ti.template(), y: ti.template(), z: ti.template()): 
	for i in range(M):
		for j in range(N):
			r[i,j] = ti.sqrt(x[i,j]**2 + y[i,j]**2 + z[i,j]**2)

@ti.kernel
def populateN(M: int, N: int, n: ti.template(), r: ti.template(), h: float):
	for i in range(M):
		for j in range(N):
			n[i, j] = -2 * ti.pow(math.e, (-r[i,j]**2 / h**2)) / (h**5) / ((math.pi)**(3/2))

@ti.kernel
def populateW(M: int, N: int, wx: ti.template(), wy: ti.template(), wz: ti.template(), n: ti.template(), x: ti.template(), y: ti.template(), z: ti.template()):
	for i in range(M):
		for j in range(N):
			wx[i, j] = n[i,j] * x[i,j]
			wy[i, j] = n[i,j] * y[i,j]
			wz[i, j] = n[i,j] * z[i,j]

def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""

	M, N = x.shape
	r = ti.field(float, shape=(M,N))
	populateR(M, N, r, x, y, z)
	

	n = ti.field(float, shape=(M,N))
	populateN(M, N, n, r, h)
	print(n)

	wx = ti.field(float, shape=(M,N))
	wy = ti.field(float, shape=(M,N))
	wz = ti.field(float, shape=(M,N))
	populateW(M, N, wx, wy, wz, n, x, y, z)
	
	return wx, wy, wz
	

def getPairwiseSeparations(ri: ti.template(), rj: ti.template()):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	M = ri.shape[0]
	N = rj.shape[0]


	rix = ti.field(float, shape=M)
	riy = ti.field(float, shape=M)
	riz = ti.field(float, shape=M)

	for i in range(M):
		rix[i] = ri[i].x
		riy[i] = ri[i].y
		riz[i] = ri[i].z
	
	# other set of points positions rj = (x,y,z)
	rjx = ti.field(float, shape=N)
	rjy = ti.field(float, shape=N)
	rjz = ti.field(float, shape=N)
	for j in range(N):
		rjx[j] = rj[j].x
		rjy[j] = rj[j].y
		rjz[j] = rj[j].z
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = ti.field(float, shape=(M,N))
	dy = ti.field(float, shape=(M,N))
	dz = ti.field(float, shape=(M,N))
	for i in range(M):
		for j in range(N):
			dx[i, j] = rix[i] - rjx[j]
			dy[i, j] = riy[i] - rjy[j]
			dz[i, j] = riz[i] - rjz[j]
	return dx, dy, dz
	

def getDensity( r, pos, m, h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of accelerations
	"""
	M = r.shape[0]
	N = pos.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos )
	rho = ti.field(float, shape=(M, 1))
	res = W( dx, dy, dz, h )
	for i in range(M):
		for j in range(N):
			rho[i, 0] = m * res[i,j]
		
	#rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
	
	return rho
	
	
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	P = ti.field(float, shape=rho.shape)
	for i in range(rho.shape[0]):
		P[i,0] = k * rho[i,0]**n
	
	return P
	

def getAcc( pos, vel, m, h, k, n, lmbda, nu ):
	"""
	Calculate the acceleration on each SPH particle
	pos   is an N x 3 matrix of positions
	vel   is an N x 3 matrix of velocities
	m     is the particle mass
	h     is the smoothing length
	k     equation of state constant
	n     polytropic index
	lmbda external force constant
	nu    viscosity
	a     is N x 3 matrix of accelerations
	"""
	N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
	P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	dx, dy,dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	ax = ti.field(float, shape=(N,1))
	ay = ti.field(float, shape=(N,1))
	az = ti.field(float, shape=(N,1))
	for i in range(N):
		ax[N, 0] = m * ((P[i,0] / (rho[i,0]**2)) + P[i, 0]/rho[i,0]**2) * dWx[i, 0]
		ay[N, 0] = m * ((P[i,0] / (rho[i,0]**2)) + P[i, 0]/rho[i,0]**2) * dWy[i, 0]
		az[N, 0] = m * ((P[i,0] / (rho[i,0]**2)) + P[i, 0]/rho[i,0]**2) * dWz[i, 0]


	
	# pack together the acceleration components
	# Add external potential force
	a = ti.Vector.field(3, float, (N,1))
	for i in range(N):
		new_a_x = ax[i, 0] - (lmbda[0] * pos[i].x - (nu * vel[i].x))
		new_a_y = ay[i, 0] - (lmbda[1] * pos[i].y - (nu * vel[i].y))
		new_a_z = az[i, 0] - (lmbda[2] * pos[i].z - (nu * vel[i].z))
		a[i,0] = ti.Vector([new_a_x, new_a_y, new_a_z])

	
	
	return a
	


def main():
	""" N-body simulation """
	
	# Simulation parameters
	# Note: Beware settings minimum X/Y/Z values to negative, especially Y: forces like gravity will begin to work in reverse
	### ---------------------------------------------- ###
	N          = 100    # Number of particles
	t          = 0      # current time of the simulation
	dt         = 0.04   # timestep
	h          = 0.1    # smoothing length
	k          = 0.1    # equation of state constant
	n          = 2      # polytropic index
	nu         = .3     # viscosity damping
	m          = 1      # single particle mas
	g          = 9.8	# gravity
	df         = -0.9   # damping factor (negative = lose energy on collision, positive = gain energy on collision)
	external_X = 0.0    #external constant x force
	external_Y = g      #external constant y force (note positive is down)
	external_Z = 0.0    #external constant z force
	seed        = 1     #random seed for initial positions 
	max_X       = 5 	#maximum x boundary for container
	max_Y       = 5	    #maximum y boundary for container
	max_Z       = 5		#maximum z boundary for container
	min_X       = 0		#minimum x boundary for container
	min_Y       = 0		#minimum y boundary for container	
	min_Z       = 0	    #minimum z boundary for container
	### ---------------------------------------------- ###

	np.random.seed(seed) #set random seed

	lmbda = ti.Vector([[external_X, external_Y, external_Z]]) # pack external force constants into a vector
	
	pos = ti.Vector.field(3, float, shape=(N,))         # particle positions
	vel = ti.Vector.field(3, float, shape=(N,))        # particle velocities (all initialized to 0)
	pos_field = ti.Vector.field(3, float, shape=(N,))
	colors = ti.Vector.field(3, float, shape=(N,))

	# Initialize particle positions/colors randomly 
	for i in range(0, N):
		pos[i] = ti.Vector([[1, 2, 3]])
		colors[i] = ti.Vector([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)])
	

	# calculate initial gravitational accelerations
	acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )

	
	window = ti.ui.Window('N-Body', (1280, 720), vsync=True)
	canvas = window.get_canvas()
	camera = ti.ui.Camera()
	scene = ti.ui.Scene()
	
	# Simulation Main Loop
	while window.running:
		# (1/2) kick
		for i in range(N):
			vel[i].x += acc[i,0].x * dt / 2
			vel[i].y += acc[i,0].y * dt / 2
			vel[i].z += acc[i,0].z * dt / 2

			
		for i in range(N):
			pos[i].x += dt * vel[i].x
			pos[i].y += dt * vel[i].y
			pos[i].z += dt * vel[i].z
		

		# update accelerations
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		for i in range(N):
			vel[i].x += 0.5 * dt * acc[i, 0].x
			vel[i].y += 0.5 * dt * acc[i, 0].y
			vel[i].z += 0.5 * dt * acc[i, 0].z
		
		# update time
		t += dt

		# render stuff
		camera.track_user_inputs(window, hold_key=ti.ui.LMB)
		scene.ambient_light((1, 1, 1))
		scene.particles(pos_field, radius=0.05, per_vertex_color=colors)
		scene.set_camera(camera)
		canvas.scene(scene)
		window.show()
	


  
if __name__== "__main__":
  main()
