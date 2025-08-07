import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def _calculate_forces_jit(positions, masses, active, width, height, softening, G, power_law, use_mond, a0):
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    
    for i in prange(n_particles):
        if not active[i]:
            continue
            
        for j in range(i + 1, n_particles):
            if not active[j]:
                continue
            
            # Calculate distance vector
            dr_x = positions[j, 0] - positions[i, 0]
            dr_y = positions[j, 1] - positions[i, 1]
            
            # Apply periodic boundary conditions
            dr_x = dr_x - width * np.round(dr_x / width)
            dr_y = dr_y - height * np.round(dr_y / height)
            
            # Distance with softening
            r2 = dr_x**2 + dr_y**2 + softening**2
            r = np.sqrt(r2)
            
            # Calculate force magnitude based on power law
            if use_mond:
                # MOND modification for low accelerations
                a_newton = G * masses[j] / r2
                if a_newton < a0:
                    # Deep MOND regime - simplified nu function
                    nu_val = 0.5 * (1 + np.sqrt(1 + 4 * a0 / a_newton))
                    nu_norm = nu_val / 1.618033988749895  # nu(1) precalculated
                    F_mag = G * masses[i] * masses[j] / (nu_norm * r2)
                else:
                    # Newtonian regime
                    F_mag = G * masses[i] * masses[j] / r2
            else:
                # General power law: F ~ r^d
                F_mag = G * masses[i] * masses[j] * r**(power_law)
            
            # Force vector components
            F_x = F_mag * dr_x / r
            F_y = F_mag * dr_y / r
            
            # Apply equal and opposite forces
            forces[i, 0] += F_x
            forces[i, 1] += F_y
            forces[j, 0] -= F_x
            forces[j, 1] -= F_y
            
    return forces

@njit(cache=True)
def _check_collisions_jit(positions, velocities, masses, active, width, height, collision_radiuses):
    n_particles = len(positions)
    
    for i in range(n_particles):
        if not active[i]:
            continue
            
        for j in range(i + 1, n_particles):
            if not active[j]:
                continue
            
            dr_x = positions[j, 0] - positions[i, 0]
            dr_y = positions[j, 1] - positions[i, 1]
            dr_x = dr_x - width * np.round(dr_x / width)
            dr_y = dr_y - height * np.round(dr_y / height)
            
            if np.sqrt(dr_x**2 + dr_y**2) < (collision_radiuses[i, 0] + collision_radiuses[j, 0]):
                # Merge particles: conserve momentum and mass
                total_mass = masses[i] + masses[j]
                positions[i, 0] = (masses[i] * positions[i, 0] + masses[j] * positions[j, 0]) / total_mass
                positions[i, 1] = (masses[i] * positions[i, 1] + masses[j] * positions[j, 1]) / total_mass
                velocities[i, 0] = (masses[i] * velocities[i, 0] + masses[j] * velocities[j, 0]) / total_mass
                velocities[i, 1] = (masses[i] * velocities[i, 1] + masses[j] * velocities[j, 1]) / total_mass
                masses[i] = total_mass
                active[j] = False

class Sim:
    def __init__(self, 
                 width=1e3, 
                 height=1e3, 
                 n_particles=1e2, 
                 power_law=-1.0, 
                 G=5.0, 
                 a0=1.0,
                 softening=0.5, 
                 dt=0.01,
                 positions=None,
                 velocities=None,
                 masses=None,
                 active=None):
        """
        Initialize gravity simulation with variable power law.
        
        Parameters:
        - width, height: grid dimensions
        - n_particles: number of particles
        - power_law: exponent d in F~r^d (default -2 for Newton)
        - G: gravitational constant
        - softening: prevents singularities at small distances
        - dt: time step
        """
        self.width = width
        self.height = height
        self.scale = (self.width + self.height) / 2 / 100
        self.n_particles = n_particles
        self.power_law = power_law
        self.G = G
        self.softening = softening
        self.dt = dt
        
        # Initialize particles
        if positions is None:
            self.positions = np.random.normal(loc=np.array([width / 2, height / 2]), 
                                              scale=np.array([width / 8, height / 8]), 
                                              size=(n_particles, 2))
        else:
            self.positions = positions
        if velocities is None:
            self.velocities = np.random.normal(loc=0.0, 
                                               scale=3 * self.scale, 
                                               size=(n_particles, 2))
        else:
            self.velocities = velocities
        if masses is None:
            self.masses = np.random.normal(loc=1.0, 
                                           scale=2.0, 
                                           size=n_particles)
        else:
            self.masses = masses
        if active is None:
            self.active = np.ones(n_particles, 
                                  dtype=bool)
        else:
            self.active = active
        
        # For MOND-like modifications
        self.a0 = a0  # MOND acceleration scale
        self.use_mond = False

    def nu(self, x):
        # Simple MOND fit to Newton's law
        # F_MOND = GmM / (nu(a0/a) * r2)
        if np.abs(1/x) > 1e-10:
            return 0.5 * (1 + np.sqrt(1 + 4 * x))
        else: 
            return 1e10
    
    def nu_norm(self, x):
        # Ensures continuos transition between Newton's law and F_MOND at x = 1
        return self.nu(x) / self.nu(1)
        
    def calculate_forces(self):
        """Calculate forces between all particle pairs."""
        return _calculate_forces_jit(
            self.positions, self.masses, self.active,
            self.width, self.height, self.softening,
            self.G, self.power_law, self.use_mond, self.a0
        )
    
    def get_particle_radii(self, masses=None):
        """Calculate visual radius for each particle based on its mass.
        
        Parameters:
        - masses: Optional mass array, uses self.masses if not provided
        
        Returns:
        - Array of visual radii for rendering
        """
        if masses is None:
            masses = self.masses
        base_size = (self.width + self.height) / 2 * 0.005
        return np.maximum(1, np.sqrt(np.abs(masses)) * base_size)
    
    def check_collisions(self):
        """Check for particle collisions and merge them."""
        # Get visual radii and scale down by 0.9 for collision detection
        visual_radii = self.get_particle_radii()
        collision_radiuses = (0.33 * visual_radii).reshape(-1, 1)
        _check_collisions_jit(
            self.positions, self.velocities, self.masses, self.active,
            self.width, self.height, collision_radiuses
        )
    
    def update(self, cached_forces=None):
        """Update particle positions using leapfrog integration.
        
        Parameters:
        - cached_forces: Optional pre-computed forces to use for first half-step
        """
        # Use cached forces or calculate new ones
        if cached_forces is None:
            forces = self.calculate_forces()
        else:
            forces = cached_forces
        
        # Update velocities (half step)
        accelerations = forces / self.masses[:, np.newaxis]
        accelerations[~self.active] = 0
        self.velocities += accelerations * self.dt * 0.5
        
        # Update positions
        self.positions[self.active] += self.velocities[self.active] * self.dt
        
        # Apply periodic boundary conditions
        self.positions %= np.array([self.width, self.height])
        
        # Recalculate forces at new positions
        forces_new = self.calculate_forces()
        
        # Update velocities (second half step)
        accelerations = forces_new / self.masses[:, np.newaxis]
        accelerations[~self.active] = 0
        self.velocities += accelerations * self.dt * 0.5
        
        # Check for collisions
        self.check_collisions()
        
        # Return new forces for potential caching in next step
        return forces_new
    
    def get_kinetic_energy(self):
        """Calculate total kinetic energy."""
        return 0.5 * np.sum(self.masses[self.active] * 
                           np.sum(self.velocities[self.active]**2, axis=1))
    
    def get_potential_energy(self):
        """Calculate total potential energy."""
        U = 0
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            for j in range(i + 1, self.n_particles):
                if not self.active[j]:
                    continue
                
                dr = self.positions[j] - self.positions[i]
                dr[0] = dr[0] - self.width * np.round(dr[0] / self.width)
                dr[1] = dr[1] - self.height * np.round(dr[1] / self.height)
                
                r = np.sqrt(np.sum(dr**2) + self.softening**2)
                
                if self.power_law == -1:
                    U -= self.G * self.masses[i] * self.masses[j] * np.log(r)
                else:
                    U -= self.G * self.masses[i] * self.masses[j] * r**(self.power_law + 1) / (self.power_law + 1)
        
        return U
    
    def get_state(self, fields=None):
        """Get current simulation state.
        
        Parameters:
        - fields: List of fields to include. If None, includes all.
                 Options: 'positions', 'velocities', 'masses', 'active'
        """
        if fields is None:
            return {
                'positions': self.positions.copy(),
                'velocities': self.velocities.copy(),
                'masses': self.masses.copy(),
                'active': self.active.copy()
            }
        
        state = {}
        if 'positions' in fields:
            state['positions'] = self.positions[self.active].copy()
        if 'velocities' in fields:
            state['velocities'] = self.velocities[self.active].copy()
        if 'masses' in fields:
            state['masses'] = self.masses[self.active].copy()
        if 'active' in fields:
            state['active'] = self.active.copy()
        return state
    
    def pre_simulate(self, n_steps, fields=None, callback=None, use_force_cache=True):
        """Pre-simulate n_steps and return trajectory.
        
        Parameters:
        - n_steps: Number of steps to simulate
        - fields: Fields to store per frame (default: positions and masses)
        - callback: Optional function called each step with (step, state)
        - use_force_cache: Whether to cache forces between steps
        """
        if fields is None:
            fields = ['positions', 'masses']
        
        trajectory = []
        cached_forces = None
        
        for step in range(n_steps):
            # Run simulation step with optional force caching
            if use_force_cache:
                cached_forces = self.update(cached_forces)
            else:
                self.update()
            
            # Store state
            state = self.get_state(fields)
            state['step'] = step
            trajectory.append(state)
            
            # Optional callback for progress reporting
            if callback:
                callback(step, state)
        
        return trajectory

