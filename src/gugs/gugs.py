import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from .gravity_simulation import GravitySimulation


class GUGS:
    def __init__(self, 
                 text="GUGS",
                 fill_rate=1.0,
                 width=800, 
                 height=600, 
                 n_particles=500,
                 power_law=-1.0,
                 G=5.0,
                 a0=1.0,
                 softening=0.5,
                 dt=0.01,
                 simulation_speed=1.0,
                 gif_duration=10.0,
                 fps=30):
        """
        Initialize GUGS (Github Username Gravity Simulation).
        
        Parameters:
        - text: Text to display using particles
        - width, height: Canvas dimensions
        - n_particles: Number of particles
        - power_law: Gravity power law exponent
        - G: Gravitational constant
        - a0: MOND acceleration scale
        - softening: Softening parameter
        - dt: Time step
        - simulation_speed: How fast the simulation runs in simulation time
        - gif_duration: Duration of the GIF in seconds
        - fps: Frames per second for the GIF
        """
        self.text = text
        self.w = width
        self.h = height
        self.n = n_particles
        self.simulation_speed = simulation_speed
        self.gif_duration = gif_duration
        self.fps = fps
        
        # Generate initial positions from text pattern
        positions = self._create_exact_text_pattern(text, fill_rate)
        
        # Initialize the gravity simulation
        self.sim = GravitySimulation(
            width=width,
            height=height,
            n_particles=n_particles,
            power_law=power_law,
            G=G,
            a0=a0,
            softening=softening,
            dt=dt * simulation_speed,  # Apply simulation speed to time step
            positions=positions,
            velocities=np.zeros((n_particles, 2)),  # Start with zero velocity
            masses=np.random.normal(loc=1.0, 
                                    scale=0.2, 
                                    size=n_particles),
            active=None   # All particles active by default
        )
        
        # Store total mass for colormap scaling
        self.total_mass = np.max(self.sim.masses)
        
    def _create_exact_text_pattern(self, username, fill_rate) -> np.ndarray:
        """Create exact text pattern from username using PIL"""
        # Create image for text rendering
        img = Image.new('L', (self.w, self.h), 0)
        draw = ImageDraw.Draw(img)
        
        # Calculate font size to make text span 80% of width
        desired_text_width = self.w * 0.8
        
        # For Liberation Mono Bold, character width is approximately 0.6 * font_size
        # So: text_width = num_chars * char_width = num_chars * 0.6 * font_size
        # Solving for font_size: font_size = text_width / (num_chars * 0.6)
        char_width_ratio = 0.6  # Typical for monospace bold fonts
        font_size = int(desired_text_width / (len(username) * char_width_ratio))
        
        # Use Liberation Mono font from repo
        import os
        package_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(package_dir, "..", "..", "liberation-mono", "LiberationMono-Regular.ttf")
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), username, font=font)
        text_width = bbox[2] - bbox[0]
        
        print(f"Using font: {font_path} at size {font_size}")
        print(f"Text width: {text_width} pixels ({text_width/self.w*100:.1f}% of canvas width)")
        
        # Get final text dimensions
        bbox = draw.textbbox((0, 0), username, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (self.w - text_width) // 2
        y = (self.h - text_height) // 2
        
        # Draw text
        draw.text((x, y), username, fill=255, font=font)
        
        # Convert to numpy array
        text_array = np.array(img)
        
        # Find all lit pixels
        lit_pixels = np.argwhere(text_array > 128)
        
        if len(lit_pixels) == 0:
            # Fallback pattern if text rendering fails
            return self._create_fallback_pattern()
        
        # Create particle positions from lit pixels
        # Limit particles to fill_rate of available pixels to avoid collisions
        max_particles = int(fill_rate * len(lit_pixels))
        
        if self.n > max_particles:
            print(f"Warning: Requested {self.n} particles but only {len(lit_pixels)} pixels available.")
            print(f"Using {max_particles} particles (90% of pixels) to avoid collisions.")
            actual_n = max_particles
        else:
            actual_n = self.n
        
        # Sample from lit pixels
        indices = np.random.choice(len(lit_pixels), actual_n, replace=False)
        pixel_positions = lit_pixels[indices]
        
        # Update particle count if it was reduced
        if actual_n < self.n:
            self.n = actual_n
            self.sim.n_particles = actual_n
        
        # Convert to x,y coordinates (swap indices and flip y-axis)
        positions = pixel_positions[:, [1, 0]].astype(np.float32)
        positions[:, 1] = self.h - positions[:, 1]  # Flip y-axis to correct orientation
        
        return positions
    
    def _create_fallback_pattern(self) -> np.ndarray:
        """Create a fallback pattern if text rendering fails"""
        # Simple circular pattern
        angles = np.linspace(0, 2 * np.pi, self.n)
        radius = min(self.w, self.h) / 3
        center = np.array([self.w / 2, self.h / 2])
        
        positions = np.zeros((self.n, 2))
        for i, angle in enumerate(angles):
            positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
            
        return positions
    
    def generate_gif(self, filename="simulation.gif"):
        """Generate a GIF of the simulation"""
        # Calculate number of frames
        n_frames = int(self.gif_duration * self.fps)
        steps_per_frame = max(1, int(1.0 / (self.fps * self.sim.dt)))
        
        print(f"Generating GIF with {n_frames} frames")
        print(f"Simulation speed: {self.simulation_speed}x")
        print(f"Steps per frame: {steps_per_frame}")
        print(f"Total simulation time: {n_frames * steps_per_frame * self.sim.dt:.2f} units")
        
        # Set up the figure with no padding
        # Calculate figure size in inches based on desired pixel dimensions
        dpi = 100
        fig_width = self.w / dpi
        fig_height = self.h / dpi
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='#0B0C1A')
        ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
        
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect('equal')
        ax.set_facecolor('#0B0C1A')  # Very dark blue background
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Turn off axis completely
        
        # Create scatter plot with plasma colormap
        scatter = ax.scatter([], [], c=[], s=[], edgecolors='none', cmap='spring', vmin=0, vmax=self.total_mass)
        
        def init():
            return scatter,
        
        def animate(frame):
            # Update simulation multiple times per frame for smoother motion
            for _ in range(steps_per_frame):
                self.sim.update()
            
            # Get active particles
            active_positions = self.sim.positions[self.sim.active]
            active_masses = self.sim.masses[self.sim.active]
            
            # Update scatter plot
            scatter.set_offsets(active_positions)
            scatter.set_array(active_masses)  # Color based on mass
            scatter.set_sizes(20 * np.sqrt(np.abs(active_masses)))  # Size based on mass
            
            # Log progress
            if frame % 10 == 0:
                print(f"Frame {frame}/{n_frames} - Active particles: {np.sum(self.sim.active)}")
            
            return scatter,
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=n_frames, interval=1000/self.fps, blit=True)
        
        # Save as GIF
        print(f"Saving GIF to {filename}...")
        writer = PillowWriter(fps=self.fps)
        anim.save(filename, writer=writer)
        plt.close(fig)
        
        print("GIF saved successfully!")
        
        return anim


# Example usage
if __name__ == "__main__":
    import sys
    
    # Get username from command line argument or use default
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = "GUGS"
        print(f"No username provided. Using default: {username}")
        print("Usage: python gugs.py <username>")
    
    # Create GUGS simulation
    gugs = GUGS(
        text=username,
        width=800,
        height=600,
        n_particles=200,
        power_law=-1.0,
        G=10.0,
        dt=0.01,
        simulation_speed=2.0,
        gif_duration=20.0,
        fps=10
    )
    
    # Generate GIF
    output_filename = "current.gif"
    gugs.generate_gif(output_filename)
    print(f"Generated: {output_filename}")