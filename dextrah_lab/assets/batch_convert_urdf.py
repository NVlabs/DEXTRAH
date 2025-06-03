# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``omni.isaac.urdf_importer``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument("input", type=str, help="The path to the input URDF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

def update_urdf(urdf_full_path, object_name):
    # Read the contents of the file
    with open(urdf_full_path, 'r') as file:
        lines = file.readlines()

    # Process the lines and replace the matching line
    new_lines = []
    search_string = '<robot name='
    replacement_string = '<robot name="object_' + object_name + '">'
    print(replacement_string)
    for line in lines:
        if search_string in line:
            new_lines.append(replacement_string + '\n')  # Append the replacement string
        else:
            new_lines.append(line)  # Append the original line

    # Write the modified contents back to the file
    with open(urdf_full_path, 'w') as file:
        file.writelines(new_lines)

def main():
    """
    Example usage:
    python batch_convert_urdf.py /home/karl/dev/dextrah_lab/dextrah_lab/assets/visdex_objects/urdf /home/karl/dev/dextrah_lab/dextrah_lab/assets/visdex_objects/USD
    """
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    usd_path = args_cli.output
    if not os.path.isabs(usd_path):
        usd_path = os.path.abspath(usd_path)

    # List all subdirectories in the target directory
    sub_dirs = os.listdir(urdf_path)

    # Filter out all subdirectories deeper than one level
    sub_dirs = [object_name for object_name in sub_dirs if os.path.isdir(os.path.join(urdf_path, object_name))]

    # Create Urdf converter config
    for object_name in sub_dirs:
        full_object_urdf_path = urdf_path + "/" + object_name + "/model.urdf"
        full_object_usd_path = usd_path + "/" + object_name
        usd_filename = object_name + ".usd"

        update_urdf(full_object_urdf_path, object_name)

        # TODO: make this a CLI arg
        convex_decompose_mesh = True

        urdf_converter_cfg = UrdfConverterCfg(
            asset_path=full_object_urdf_path,
            usd_dir=full_object_usd_path,
            usd_file_name= usd_filename,
            fix_base=args_cli.fix_base,
            merge_fixed_joints=args_cli.merge_joints,
            force_usd_conversion=True,
            make_instanceable=args_cli.make_instanceable,
            convex_decompose_mesh=convex_decompose_mesh,
        )

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input URDF file: {urdf_path}")
        print("URDF importer config:")
        print_dict(urdf_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create Urdf converter and import the file
        urdf_converter = UrdfConverter(urdf_converter_cfg)
        # print output
        print("URDF importer output:")
        print(f"Generated USD file: {urdf_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

#    # Determine if there is a GUI to update:
#    # acquire settings interface
#    carb_settings_iface = carb.settings.get_settings()
#    # read flag for whether a local GUI is enabled
#    local_gui = carb_settings_iface.get("/app/window/enabled")
#    # read flag for whether livestreaming GUI is enabled
#    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

#    # Simulate scene (if not headless)
#    if local_gui or livestream_gui:
#        # Open the stage with USD
#        stage_utils.open_stage(urdf_converter.usd_path)
#        # Reinitialize the simulation
#        app = omni.kit.app.get_app_interface()
#        # Run simulation
#        with contextlib.suppress(KeyboardInterrupt):
#            while app.is_running():
#                # perform step
#                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
