import os


def execute_scene(scene_name):
    # Check that there is scenes/{scene_name} folder
    if not os.path.exists(f'scenes/{scene_name}'):
        raise ValueError(
            f"Scene '{scene_name}' does not exist in 'scenes/' directory.")

    # move to that folder, execute `mjpython runner.py`, and then move back
    original_directory = os.getcwd()
    try:
        os.chdir(f'scenes/{scene_name}')
        os.system('mjpython runner.py')
    except Exception as e:
        raise RuntimeError(f"Failed to execute scene '{scene_name}': {e}")
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python execute_scene.py <scene_name>")
        sys.exit(1)

    scene_name = sys.argv[1]
    try:
        execute_scene(scene_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
