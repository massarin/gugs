import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
from .sim import Sim


class GUGS:
    def __init__(self, 
                 text="GUGS",
                 fill_rate=1.0,
                 width=800, 
                 height=600, 
                 n_particles=500,
                 power_law=-1.0,
                 G=3.0,
                 a0=1.0,
                 softening=0.5,
                 dt=0.005,
                 simulation_speed=1.0,
                 gif_duration=10.0,
                 fps=30,
                 reverse=False,
                 reverse_duration=5.0):
        """
        Initialize GUGS.
        
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
        self.reverse = reverse
        self.reverse_duration = reverse_duration
        
        # Generate initial positions from text pattern
        positions = self._create_exact_text_pattern(text, fill_rate)
        
        # Initialize the gravity simulation
        self.sim = Sim(
            width=width,
            height=height,
            n_particles=n_particles,
            power_law=power_law,
            G=G,
            a0=a0,
            softening=softening,
            dt=dt * simulation_speed,  # Apply simulation speed to time step
            positions=positions,
            velocities=np.random.normal(loc=0.0, 
                                        scale=0.5 * (self.w + self.h)/200, 
                                        size=(n_particles, 2)),
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
        
        # Convert to x,y coordinates (swap indices)
        positions = pixel_positions[:, [1, 0]].astype(np.float32)
        
        return positions
    
    def _create_reverse_frames(self, trajectory):
        """Create reverse frames with accelerating speed.
        
        The reverse playback starts slow and accelerates exponentially.
        """
        # Total frames for reverse playback at target fps
        reverse_frame_count = int(self.reverse_duration * self.fps)
        
        # We'll sample from the forward trajectory with increasing gaps
        # Use exponential acceleration: frame_index = exp(t) - 1
        forward_frames = len(trajectory)
        
        # Slow down the last few frames of forward playback first
        # Add 5 frames that repeat the last frame with slight variations
        slowdown_frames = []
        last_frame = trajectory[-1]
        for _ in range(5):
            slowdown_frames.append(last_frame)
        
        # Now create the reverse frames with acceleration
        reverse_frames = []
        
        # We want to map reverse_frame_count frames to the trajectory
        # Using an exponential curve for sampling
        for i in range(reverse_frame_count):
            # Progress from 0 to 1
            progress = i / max(1, reverse_frame_count - 1)
            
            # Use exponential curve: starts slow, accelerates
            # Map progress exponentially from end to beginning
            exp_progress = (np.exp(progress * 3) - 1) / (np.exp(3) - 1)
            
            # Map to trajectory index (reversed)
            traj_index = int((1 - exp_progress) * (forward_frames - 1))
            traj_index = max(0, min(forward_frames - 1, traj_index))
            
            reverse_frames.append(trajectory[traj_index])
        
        # Combine slowdown and reverse
        return slowdown_frames + reverse_frames
    
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
    
    def mass_to_color(self, mass, max_mass):
        """Convert mass to RGB color using spring colormap."""
        # Normalize mass to [0, 1]
        norm = mass / max_mass
        # Spring colormap: magenta (1,0,1) -> yellow (1,1,0)
        r = 1.0
        g = norm
        b = 1.0 - norm
        return tuple(int(c * 255) for c in [r, g, b])
    
    def render_frame_pil(self, positions, masses, width=None, height=None):
        """Render a frame using PIL for faster performance.
        
        Parameters:
        - positions: Array of particle positions
        - masses: Array of particle masses
        - width, height: Canvas dimensions (uses self.w, self.h if not provided)
        """
        if width is None:
            width = self.w
        if height is None:
            height = self.h
            
        # Create image with dark background
        img = Image.new('RGB', (width, height), '#0B0C1A')
        draw = ImageDraw.Draw(img)
        
        # Draw particles
        max_mass = np.max(masses) if len(masses) > 0 else 1.0
        # Get particle radii from simulation (single source of truth)
        radii = self.sim.get_particle_radii(masses)
        for pos, mass, radius in zip(positions, masses, radii):
            # Convert to integer for drawing
            radius = int(radius)
            
            # Get color from spring colormap
            color = self.mass_to_color(mass, max_mass)
            
            # Draw particle as filled circle
            x, y = int(pos[0]), int(pos[1])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline=None)
        
        return img
    
    def pre_simulate_trajectory(self, n_frames, steps_per_frame=1, verbose=True):
        """Pre-simulate the entire trajectory.
        
        Parameters:
        - n_frames: Number of frames to generate
        - steps_per_frame: Simulation steps per frame
        - verbose: Print progress
        
        Returns:
        - trajectory: List of states for each frame
        """
        total_steps = n_frames * steps_per_frame
        
        if verbose:
            print(f"Pre-simulating {total_steps} steps ({n_frames} frames)...")
        
        def progress_callback(step, state):
            if verbose and step % (10 * steps_per_frame) == 0:
                frame = step // steps_per_frame
                active_count = np.sum(self.sim.active)
                print(f"Frame {frame}/{n_frames} - Active particles: {active_count}")
        
        # Run pre-simulation with force caching
        trajectory_raw = self.sim.pre_simulate(
            total_steps, 
            fields=['positions', 'masses'],
            callback=progress_callback if verbose else None,
            use_force_cache=True
        )
        
        # Sample frames from trajectory
        trajectory = [trajectory_raw[i * steps_per_frame] 
                     for i in range(n_frames)]
        
        if verbose:
            print("Pre-simulation complete!")
        
        return trajectory
    
    def generate_gif_from_trajectory(self, trajectory, filename="simulation.gif"):
        """Generate GIF from pre-computed trajectory.
        
        Parameters:
        - trajectory: List of states from pre_simulate_trajectory
        - filename: Output filename
        """
        frames_to_write = trajectory.copy()
        
        if self.reverse:
            # Add reverse playback with accelerating speed
            reverse_frames = self._create_reverse_frames(trajectory)
            frames_to_write.extend(reverse_frames)
        
        n_frames = len(frames_to_write)
        print(f"Generating GIF with {n_frames} frames using PIL...")
        
        # Use imageio with PIL rendering
        with imageio.get_writer(filename, mode='I', fps=self.fps) as writer:
            for i, state in enumerate(frames_to_write):
                # Render frame with PIL
                img = self.render_frame_pil(state['positions'], state['masses'])
                writer.append_data(np.array(img))
                
                if i % 10 == 0:
                    print(f"Rendered frame {i}/{n_frames}")
        
        print(f"GIF saved to {filename}!")
    
    def generate_gif(self, filename="simulation.gif"):
        """Generate a GIF of the simulation.
        
        Parameters:
        - filename: Output filename
        """
        # Calculate number of frames
        n_frames = int(self.gif_duration * self.fps)
        steps_per_frame = max(1, int(1.0 / (self.fps * self.sim.dt)))
        
        print(f"Generating GIF with {n_frames} frames")
        print(f"Simulation speed: {self.simulation_speed}x")
        print(f"Steps per frame: {steps_per_frame}")
        print(f"Total simulation time: {n_frames * steps_per_frame * self.sim.dt:.2f} units")
        
        # Pre-simulate trajectory
        trajectory = self.pre_simulate_trajectory(n_frames, steps_per_frame)
        
        # Generate GIF from trajectory
        self.generate_gif_from_trajectory(trajectory, filename)
        
        return trajectory