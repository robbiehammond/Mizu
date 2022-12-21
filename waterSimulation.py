import numpy as np
from scipy.special import gamma
import taichi as ti
ti.init(arch=ti.cpu)

def W( x, y, z, h ):
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
	
def gradW( x, y, z, h ):
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T
	
	return dx, dy, dz
	

def getDensity( r, pos, m, h ):
	M = r.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos )
	
	rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
	
	return rho
	
	
def getPressure(rho, k, n):
	P = k * rho**(1+1/n)
	
	return P
	

def getAcc( pos, vel, m, h, k, n, lmbda, nu ):
	N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
	P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	dx, dy,dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
	ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
	az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))
	
	# Add external potential force
	a -= lmbda * pos
	
	# Add viscosity
	a -= nu * vel
	
	return a
	


def main():
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
	df         = -0.7   # damping factor
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

	lmbda = np.array([[external_X, external_Y, external_Z]]) # pack external force constants into a vector
	
	pos = np.zeros(shape=(N, 3))         # particle positions
	vel = np.zeros(pos.shape)        # particle velocities (all initialized to 0)
	pos_field = ti.Vector.field(3, float, shape=(N,))
	colors = ti.Vector.field(3, float, shape=(N,))

	# Initialize particle positions/colors randomly 
	for i in range(0, N):
		pos[i] = np.array([[np.random.uniform(min_X, max_X), np.random.uniform(min_Y, max_Y), np.random.uniform(min_Z, max_Z) ]])
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
		vel += acc * dt/2
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		vel += acc * dt/2
		
		# update time
		t += dt

		# if particle is at exceeds, reverse its velocity, dampen it a bit, and put it back in the bounding container
		out_of_left_boundary = pos[:, 0] < min_X
		vel[out_of_left_boundary, 0] *= df
		pos[out_of_left_boundary, 0] = min_X

		out_of_right_boundary = pos[:, 0] > max_X
		vel[out_of_right_boundary, 0] *= df
		pos[out_of_right_boundary, 0] = max_X

		out_of_bottom_boundary = pos[:, 1] < min_Y
		vel[out_of_bottom_boundary, 1] *= df
		pos[out_of_bottom_boundary, 1] = min_Y

		out_of_top_boundary = pos[:, 1] > max_Y
		vel[out_of_top_boundary, 1] *= df
		pos[out_of_top_boundary, 1] = max_Y

		out_of_back_boundary = pos[:, 2] < min_Z
		vel[out_of_back_boundary, 2] *= df
		pos[out_of_back_boundary, 2] = min_Z

		out_of_front_boundary = pos[:, 2] > max_Z
		vel[out_of_front_boundary, 2] *= df
		pos[out_of_front_boundary, 2] = max_Z


		# render stuff
		pos_field.from_numpy(pos)
		camera.track_user_inputs(window, hold_key=ti.ui.LMB)
		scene.ambient_light((1, 1, 1))
		scene.particles(pos_field, radius=0.05, per_vertex_color=colors)
		scene.set_camera(camera)
		canvas.scene(scene)
		window.show()
	


  
if __name__== "__main__":
  main()
