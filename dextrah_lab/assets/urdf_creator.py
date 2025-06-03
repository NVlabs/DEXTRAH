import os
import shutil

def generate_urdf(target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        print(f"Target directory {target_dir} does not exist.")
        return

    # Crawl the directory for STL pairs
    stl_files = [f for f in os.listdir(target_dir) if f.endswith('.stl')]
    stl_pairs = {}

    for stl in stl_files:
        if stl.endswith('_simple.stl'):
            object_name = stl.replace('_simple.stl', '')
            if object_name in stl_pairs:
                stl_pairs[object_name][1] = stl
            else:
                stl_pairs[object_name] = [None, stl]
        else:
            object_name = stl.replace('.stl', '')
            if object_name in stl_pairs:
                stl_pairs[object_name][0] = stl
            else:
                stl_pairs[object_name] = [stl, None]

    for object_name, pair in stl_pairs.items():
        if pair[0] and pair[1]:  # Ensure both files exist for the pair
            original_stl, simple_stl = pair

            # Create subdirectory for the object
            object_dir = os.path.join(target_dir, object_name)
            os.makedirs(object_dir, exist_ok=True)

            # Move the STL files to the subdirectory
            shutil.move(os.path.join(target_dir, original_stl), os.path.join(object_dir, original_stl))
            shutil.move(os.path.join(target_dir, simple_stl), os.path.join(object_dir, simple_stl))

            # Generate the URDF content
            urdf_content = f"""<?xml version='1.0'?>

<robot name="{object_name}">
    <static>false</static>
    <link name="baseLink">
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1e-4" ixy="0." ixz="0." iyy="1e-4" iyz="0." izz="1e-4"/>
        </inertial>
        <collision name="collision">
            <geometry>
                <mesh filename="{original_stl}"/>
            </geometry>
        </collision>
        <visual name="visual">
            <geometry>
                <mesh filename="{simple_stl}"/>
            </geometry>
        </visual>
    </link>
</robot>"""

            # Write the URDF to the subdirectory
            urdf_path = os.path.join(object_dir, "model.urdf")
            with open(urdf_path, 'w') as urdf_file:
                urdf_file.write(urdf_content)

if __name__ == "__main__":
    target_directory = input("Enter the target directory: ").strip()
    generate_urdf(target_directory)
