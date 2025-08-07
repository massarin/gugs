import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GravitySimulation:
    def __init__(self, 
                 width=800, 
                 height=600, 
                 n_particles=100, 
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
        self.collision_radius = self.scale
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
        forces = np.zeros_like(self.positions)
        
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
                
            for j in range(i + 1, self.n_particles):
                if not self.active[j]:
                    continue
                
                # Calculate distance vector
                dr = self.positions[j] - self.positions[i]
                
                # Apply periodic boundary conditions
                dr[0] = dr[0] - self.width * np.round(dr[0] / self.width)
                dr[1] = dr[1] - self.height * np.round(dr[1] / self.height)
                
                # Distance with softening
                r2 = np.sum(dr**2) + self.softening**2
                r = np.sqrt(r2)
                
                # Calculate force magnitude based on power law
                if self.use_mond:
                    # MOND modification for low accelerations
                    a_newton = self.G * self.masses[j] / r2
                    if a_newton < self.a0:
                        # Deep MOND regime
                        F_mag = self.G * self.masses[i] * self.masses[j] / (self.nu_norm(self.a0 / a_newton) * r2)
                    else:
                        # Newtonian regime
                        F_mag = self.G * self.masses[i] * self.masses[j] / r2
                else:
                    # General power law: F ~ r^d
                    F_mag = self.G * self.masses[i] * self.masses[j] * r**(self.power_law)
                
                # Force vector
                F_vec = F_mag * dr / r
                
                # Apply equal and opposite forces
                forces[i] += F_vec
                forces[j] -= F_vec
                
        return forces
    
    def check_collisions(self):
        """Check for particle collisions and merge them."""
            
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
                
            for j in range(i + 1, self.n_particles):
                if not self.active[j]:
                    continue
                
                dr = self.positions[j] - self.positions[i]
                dr[0] = dr[0] - self.width * np.round(dr[0] / self.width)
                dr[1] = dr[1] - self.height * np.round(dr[1] / self.height)
                
                if np.sqrt(np.sum(dr**2)) < self.collision_radius:
                    # Merge particles: conserve momentum and mass
                    total_mass = self.masses[i] + self.masses[j]
                    self.positions[i] = (self.masses[i] * self.positions[i] + 
                                       self.masses[j] * self.positions[j]) / total_mass
                    self.velocities[i] = (self.masses[i] * self.velocities[i] + 
                                        self.masses[j] * self.velocities[j]) / total_mass
                    self.masses[i] = total_mass
                    self.active[j] = False
    
    def update(self):
        """Update particle positions using leapfrog integration."""
        # Calculate forces
        forces = self.calculate_forces()
        
        # Update velocities (half step)
        accelerations = forces / self.masses[:, np.newaxis]
        accelerations[~self.active] = 0
        self.velocities += accelerations * self.dt * 0.5
        
        # Update positions
        self.positions[self.active] += self.velocities[self.active] * self.dt
        
        # Apply periodic boundary conditions
        self.positions %= np.array([self.width, self.height])
        
        # Recalculate forces at new positions
        forces = self.calculate_forces()
        
        # Update velocities (second half step)
        accelerations = forces / self.masses[:, np.newaxis]
        accelerations[~self.active] = 0
        self.velocities += accelerations * self.dt * 0.5
        
        # Check for collisions
        self.check_collisions()
    
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

def plot(width, height, n_particles, power_law, G, dt, frames, interval):
    """Single plot for testing a specific gravity law."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Create simulation with desired parameters
    sim = GravitySimulation(width=width, 
                            height=height, 
                            n_particles=n_particles,
                            power_law=power_law, 
                            G=G, 
                            dt=dt
    )
    sim.use_mond = False
    
    # Set up the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_title('Newtonian Gravity (F ~ 1/r)', color='white', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    scatter = ax.scatter([], [], c='white', s=[], edgecolors='gray')
    
    def init():
        return scatter,
    
    def animate(frame):
        sim.update()
        scatter.set_offsets(sim.positions[sim.active])
        scatter.set_sizes(20 * np.sqrt(sim.masses[sim.active]))
        return scatter,
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=frames, interval=interval, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return sim, anim


# Example usage
if __name__ == "__main__":
    sim, anim = plot(width=800, 
                     height=600, 
                     n_particles=100, 
                     power_law=-1.0, 
                     G=35, 
                     dt=0.01, 
                     frames=100, 
                     interval=10)