"""
Complete Integration Script - Run Warehouse AMR Simulation
Place this in your project root and run it directly
"""

import sys
import os

# Ensure src directory is in path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)


def check_dependencies():
    """Check if all required packages are installed."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pygame
    except ImportError:
        missing.append("pygame")

    try:
        import Box2D
    except ImportError:
        missing.append("Box2D-kiwi")

    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import skfuzzy
    except ImportError:
        missing.append("skfuzzy")

    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


def check_files():
    """Check if all required files exist."""
    required_files = [
        "src/config.py",
        "src/robot.py",
        "src/localization.py",
        "src/environment.py",
        "src/a_star.py",
        "src/dwa.py",
        "src/fuzzy_controller.py",
        "src/renderer.py",
        "src/simulation.py",
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(os.path.join(project_root, file)):
            missing.append(file)

    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False

    return True


def check_assets():
    """Check if asset directory exists."""
    assets_dir = os.path.join(project_root, "assets")

    if not os.path.exists(assets_dir):
        print("⚠️  Warning: 'assets' directory not found")
        print("   Creating basic assets...")
        create_basic_assets()
        return True

    required_assets = [
        "tile.png",
        "robot.png",
        # "obs_person_1.png",
        # "obs_robot.png",
        # "block_square.png",
    ]

    missing = []
    for asset in required_assets:
        if not os.path.exists(os.path.join(assets_dir, asset)):
            missing.append(asset)

    if missing:
        print("⚠️  Warning: Some assets are missing:")
        for asset in missing:
            print(f"   - {asset}")
        print("   Creating basic assets...")
        create_basic_assets()

    return True


def create_basic_assets():
    """Create basic placeholder assets."""
    import pygame

    assets_dir = os.path.join(project_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    pygame.init()

    # Create basic colored circle assets
    assets_config = {
        "robot.png": ((64, 64), (100, 150, 255)),
        "obs_person_1.png": ((64, 64), (255, 100, 100)),
        "obs_robot.png": ((64, 64), (150, 150, 150)),
        "tile.png": ((64, 64), (240, 240, 240)),
        "block_square.png": ((64, 64), (80, 80, 80)),
    }

    for filename, (size, color) in assets_config.items():
        surf = pygame.Surface(size, pygame.SRCALPHA)
        if "tile" in filename:
            surf.fill(color)
        else:
            pygame.draw.circle(
                surf, color, (size[0] // 2, size[1] // 2), size[0] // 2 - 2
            )

        filepath = os.path.join(assets_dir, filename)
        pygame.image.save(surf, filepath)

    pygame.quit()
    print("✅ Basic assets created in 'assets' directory")


def run_simulation():
    """Run the main simulation."""
    # Import main simulation
    try:
        from src.simulation import main

        main()

    except ImportError as e:
        print(f"❌ Error importing simulation modules: {e}")
        print("\nMake sure all files are in the correct locations:")
        print("  src/")
        print("    ├── a_star.py")
        print("    ├── config.py")
        print("    ├── dwa.py")
        print("    ├── environment.py")
        print("    ├── fuzzy_controller.py")
        print("    ├── localization.py")
        print("    ├── renderer.py")
        print("    ├── robot.py")
        print("    └── simulation.py")
        sys.exit(1)


def main():
    """Main entry point."""
    print("\n🤖 Warehouse AMR Simulation Setup\n")

    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies installed\n")

    # Check files
    print("Checking project files...")
    if not check_files():
        print(
            "\n💡 Tip: Make sure you've created all the required files from the artifacts"
        )
        sys.exit(1)
    print("✅ All required files present\n")

    # Check assets
    print("Checking assets...")
    check_assets()
    # print()

    # Run simulation
    run_simulation()


if __name__ == "__main__":
    main()
