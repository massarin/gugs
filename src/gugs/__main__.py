import sys
from .gugs import GUGS


def main():
    # Get username from command line argument or use default
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = "GUGS"
        print(f"No username provided. Using default: {username}")
        print("Usage: gugs <username>")
    
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
    output_filename = "gugs.gif"
    gugs.generate_gif(output_filename)
    print(f"Generated: {output_filename}")


if __name__ == "__main__":
    main()