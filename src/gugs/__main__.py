import sys
import argparse
from .gugs import GUGS


def main():
    parser = argparse.ArgumentParser(description="Generate a GUGS animation GIF")
    parser.add_argument("text", nargs="?", default="GUGS", help="Text to display (default: GUGS)")
    parser.add_argument("--width", type=int, default=600, help="Width of the animation (default: 600)")
    parser.add_argument("--height", type=int, default=200, help="Height of the animation (default: 200)")
    parser.add_argument("--n-particles", type=int, default=200, help="Number of particles (default: 200)")
    parser.add_argument("--power-law", type=float, default=-1.0, help="Power law exponent for initial velocities (default: -1.0)")
    parser.add_argument("--G", type=float, default=3.0, help="Gravitational constant (default: 10.0)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for simulation (default: 0.01)")
    parser.add_argument("--simulation-speed", type=float, default=2.0, help="Simulation speed multiplier (default: 2.0)")
    parser.add_argument("--gif-duration", type=float, default=20.0, help="Duration of GIF in seconds (default: 20.0)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--reverse", action="store_true", help="Add reverse playback with accelerating speed at the end")
    parser.add_argument("--reverse-duration", type=float, default=5.0, help="Duration of reverse playback in seconds (default: 5.0)")
    parser.add_argument("--output", "-o", type=str, default="gugs.gif", help="Output filename (default: gugs.gif)")
    
    args = parser.parse_args()
    
    # Create GUGS simulation
    gugs = GUGS(
        text=args.text,
        width=args.width,
        height=args.height,
        n_particles=args.n_particles,
        power_law=args.power_law,
        G=args.G,
        dt=args.dt,
        simulation_speed=args.simulation_speed,
        gif_duration=args.gif_duration,
        fps=args.fps,
        reverse=args.reverse,
        reverse_duration=args.reverse_duration
    )
    
    # Generate GIF
    gugs.generate_gif(args.output)
    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()