import xml.etree.ElementTree as ET


def get_mink_xml(scene_path):
    begin = """
    <mujoco model="ur5e scene">
      <compiler angle="radian" meshdir="assets" autolimits="true" texturedir="assets" />
      <include file="robot.xml" />

      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512"
          height="512"
          mark="cross" markrgb=".8 .8 .8" />
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" />
      </asset>

      <worldbody>
        <geom name="floor" size="1 1 0.01" type="plane" material="grid" />
"""

    def extract_environment_from_scene(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        for element in root.findall(".//*[@name='environment']"):
            return ET.tostring(element, encoding="unicode")
        return ""

    middle = extract_environment_from_scene(scene_path)

    end = """
      </worldbody>
    </mujoco>"""

    return begin + middle + end
