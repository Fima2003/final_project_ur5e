import os
from xml.etree import ElementTree as ET
import shutil


def create_scene(scene_name: str, environment_name: str, arm_name: str, gripper_name: str):
    """
    Create a scene with the specified environment, arm, and gripper.
    Checks for file existence, loads and combines XMLs, and writes connected.xml.
    """
    env_path, arm_path, gripper_path = ensure_directories_exists(
        environment_name, arm_name, gripper_name)

    create_scene_directory(scene_name)

    try:
        copy_necessary_assets(scene_name, arm_name,
                              gripper_name, environment_name)
        copy_executables(arm_name, scene_name)
        create_runner(arm_name, scene_name)

        arm_root, gripper_root, env_root = parse_xml(
            arm_path, gripper_path, env_path)

        env_worldbody = env_root.find('worldbody')

        # replace the body with name='base' to body with name='{robot_name}/base'
        env_table_height = env_worldbody.find(
            './/geom[@name="table"]').get('pos', '0 0 0.3').split(' ')[-1]
        arm_base = arm_root.find('.//body[@name="base"]')
        if arm_base is not None:
            arm_base.set('name', f"{arm_name}/base")
            arm_base.set('pos', '0 0 ' + env_table_height)

        add_gripper_to_arm(gripper_root, arm_root)

        # Change asset name of the arm and gripper to have it referred to the {arm_name}/asset and {gripper_name}/asset instead of simple asset. Then combine all assets into the arm XML.
        deal_with_assets(arm_root, gripper_root, arm_name, gripper_name)

        # find all first-level tags! not attributes of the arm and gripper XMLs
        arm_tags = [tag for tag in arm_root.findall(
            './*') if tag.tag not in ['worldbody', 'asset']]
        gripper_tags = [tag for tag in gripper_root.findall(
            './*') if tag.tag not in ['worldbody', 'asset']]

        # take the ones in gripper that are not in arm
        for tag in gripper_tags:
            if tag.tag not in [t.tag for t in arm_tags]:
                arm_root.append(tag)

        # take the ones in the gripper that are in the arm
        for tag in gripper_tags:
            if tag.tag in [t.tag for t in arm_tags]:
                arm_tag = arm_root.find(tag.tag)
                if arm_tag is not None and arm_tag.tag != 'compiler' and arm_tag.tag != 'option':
                    arm_tag.extend(tag.findall('./*'))

        # remove keyframe tags from the arm XML
        for keyframe in arm_root.findall('.//keyframe'):
            arm_root.remove(keyframe)

        for keyframe in gripper_root.findall('.//keyframe'):
            gripper_root.remove(keyframe)
        # Add environment to the scene
        env_worldbody = env_root.insert(
            0, ET.Element('include', file="robot.xml"))

        # Print the XML of arm and gripper
        robot_path = os.path.join('scenes', scene_name, "robot.xml")
        with open(robot_path, "w", encoding="utf-8") as robot_file:
            robot_file.write(ET.tostring(arm_root, encoding='unicode'))

        # Print the XML of the scene
        env_path = os.path.join('scenes', scene_name, "scene.xml")
        with open(env_path, "w", encoding="utf-8") as env_file:
            env_file.write(ET.tostring(env_root, encoding='unicode'))

    except Exception as e:
        print(f"Error creating scene: {e}")
        os.remove(os.path.join('scenes', scene_name))


def ensure_directories_exists(environment_name: str, arm_name: str, gripper_name: str):
    env_path = f"environments/{environment_name}/{environment_name}.xml"
    arm_path = f"arms/{arm_name}/{arm_name}.xml"
    gripper_path = f"grippers/{gripper_name}/{gripper_name}.xml"

    # Check files exist
    for path in [env_path, arm_path, gripper_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    return env_path, arm_path, gripper_path


def create_scene_directory(scene_name: str):
    if not os.path.exists(os.path.join('scenes', scene_name)):
        os.mkdir(os.path.join('scenes', scene_name))
    else:
        if input(f"Scene {scene_name} already exists. Do you want to overwrite it? (y/n): ") == "y":
            shutil.rmtree(os.path.join('scenes', scene_name))
            os.mkdir(os.path.join('scenes', scene_name))
        else:
            print(
                f"Scene {scene_name} already exists. Not overwriting. Quitting...")
            return
    os.mkdir(os.path.join('scenes', scene_name, "assets")) if not os.path.exists(
        os.path.join('scenes', scene_name, "assets")) else None


def copy_necessary_assets(scene_name: str, arm_name: str, gripper_name: str, environment_name: str):
    # Copy assets of arm and gripper to the scene
    arm_assets_path = os.path.join('scenes', scene_name, "assets", arm_name)
    gripper_assets_path = os.path.join(
        'scenes', scene_name, "assets", gripper_name)
    environment_assets_path = os.path.join(
        'scenes', scene_name, "assets", environment_name)
    os.mkdir(arm_assets_path) if not os.path.exists(
        arm_assets_path) else None
    os.mkdir(gripper_assets_path) if not os.path.exists(
        gripper_assets_path) else None
    os.mkdir(environment_assets_path) if not os.path.exists(
        environment_assets_path) else None
    for asset in os.listdir(f"arms/{arm_name}/assets"):
        asset_path = os.path.join(f"arms/{arm_name}/assets", asset)
        if os.path.isfile(asset_path):
            shutil.copy(asset_path, os.path.join(arm_assets_path, asset))
    for asset in os.listdir(f"grippers/{gripper_name}/assets"):
        asset_path = os.path.join(f"grippers/{gripper_name}/assets", asset)
        # copy all of the assets: files and directories
        if os.path.isfile(asset_path):
            shutil.copy(asset_path, os.path.join(
                gripper_assets_path, asset))
        elif os.path.isdir(asset_path):
            shutil.copytree(asset_path, os.path.join(
                gripper_assets_path, asset), dirs_exist_ok=True)
    for asset in os.listdir(f"environments/{environment_name}/assets"):
        asset_path = os.path.join(
            f"environments/{environment_name}/assets", asset)
        # copy all of the assets: files and directories
        if os.path.isfile(asset_path):
            shutil.copy(asset_path, os.path.join(
                environment_assets_path, asset))
        elif os.path.isdir(asset_path):
            shutil.copytree(asset_path, os.path.join(
                environment_assets_path, asset), dirs_exist_ok=True)


def copy_executables(arm_name: str, environment_name: str):
    shutil.copy(f"arms/{arm_name}/{arm_name}.py",
                f"scenes/{environment_name}/{arm_name}.py")
    shutil.copy("arms/Arm.py",
                f"scenes/{environment_name}/Arm.py")
    shutil.copy('executables/main.py', f'scenes/{environment_name}/main.py')
    shutil.copy('executables/utils.py', f'scenes/{environment_name}/utils.py')


def create_runner(arm_name: str, env_name: str):
    actual_arm_name = arm_name.upper() if arm_name != 'ur5e' else 'UR5e'
    with open(f"scenes/{env_name}/runner.py", "w") as f:
        f.write(
            f"from {arm_name} import {actual_arm_name}\n")
        f.write(f"from main import run_the_mf\n")
        f.write(f"\n")
        f.write(f"def main():\n")
        f.write(f"    robot = {actual_arm_name}()\n")
        f.write(f"    run_the_mf(robot)\n")
        f.write(f"\n")
        f.write(f"if __name__ == '__main__':\n")
        f.write(f"    main()\n")


def parse_xml(arm_path: str, gripper_path: str, env_path: str):
    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()
    gripper_tree = ET.parse(gripper_path)
    gripper_root = gripper_tree.getroot()
    env_tree = ET.parse(env_path)
    env_root = env_tree.getroot()
    return arm_root, gripper_root, env_root


def get_attachment_site(worldbody: ET.Element) -> ET.Element | None:
    attachment_site = None
    for body1 in worldbody.findall('body'):
        for site1 in body1.findall('site'):
            if site1.get('name') == 'attachment_site':
                attachment_site = site1
                break
        if attachment_site is not None:
            return attachment_site
        else:
            return get_attachment_site(body1)
    return None


def add_gripper_to_arm(gripper_root: ET.Element, arm_root: ET.Element):
    arm_worldbody = arm_root.find('worldbody')
    attachment_site = get_attachment_site(arm_worldbody)
    if attachment_site is None:
        raise ValueError("Attachment site not found in the arm XML.")
    gripper_worldbody = gripper_root.find('worldbody')
    if gripper_worldbody is None:
        raise ValueError("Gripper worldbody not found in the gripper XML.")
    gripper_body = gripper_worldbody.find('body')
    if gripper_body is None:
        raise ValueError("Gripper body not found in the gripper XML.")

    # replace attachment site with <body name="attachment"><site name="attachment_site"/></body>
    attachment_body = ET.Element(
        'body', name='attachment', pos=attachment_site.get('pos', '0 0 0'), quat=attachment_site.get('quat', '1 0 0 0'))
    attachment_body.append(
        ET.Element('site', name='attachment_site', rgba="1 0 0 1", size="0.01", group="1"))
    attachment_body.append(gripper_body)

    # Add gripper body next to the attachment site
    parent_body = arm_worldbody.find(
        f'.//site[@name="attachment_site"]...')
    if parent_body is None:
        raise ValueError("Parent body of attachment site not found.")
    parent_body.remove(attachment_site)
    parent_body.append(attachment_body)


def deal_with_assets(arm_root: ET.Element, gripper_root: ET.Element, arm_name: str, gripper_name: str):
    for asset in arm_root.findall('asset'):
        for mesh in asset.findall('mesh'):
            mesh.set('file', f"{arm_name}/{mesh.get('file')}")
    for asset in gripper_root.findall('asset'):
        for mesh in asset.findall('mesh'):
            mesh.set('file', f"{gripper_name}/{mesh.get('file')}")

    # Combine assets
    arm_asset = arm_root.find('asset')
    gripper_asset = gripper_root.find('asset')
    if arm_asset is not None and gripper_asset is not None:
        for child in gripper_asset:
            arm_asset.append(child)


if __name__ == "__main__":
    # Fetch the scene name, environment, arm, and gripper from command line arguments
    import sys
    if len(sys.argv) != 5:
        print("Creating default scene with UR5e arm and gripper.")
        scene_name = 'lab-ur5e'
        environment_name = 'lab'
        arm_name = 'ur5e'
        gripper_name = '2f85'
        # print("Usage: python create_scene.py <scene_name> <environment_name> <arm_name> <gripper_name>")
        # sys.exit(1)
    else:
        scene_name = sys.argv[1]
        environment_name = sys.argv[2]
        arm_name = sys.argv[3]
        gripper_name = sys.argv[4]
    try:
        create_scene(scene_name, environment_name, arm_name, gripper_name)
        print(f"Scene {scene_name} created successfully.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
