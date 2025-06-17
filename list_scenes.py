import os


def list_scenes():
    scenes_dir = 'scenes'
    scenes = [name for name in os.listdir(
        scenes_dir) if os.path.isdir(os.path.join(scenes_dir, name))]
    return scenes


if __name__ == "__main__":
    scenes = list_scenes()
    if scenes:
        print("Available scenes:")
        for scene in scenes:
            print(f"- {scene}")
    else:
        print("No scenes found in the 'scenes' directory.")
