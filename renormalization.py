import numpy as np

def modify_obj_vertices(input_path, output_path):
    """
    Modifies the vertices of an OBJ file by scaling them by 0.5 and then shifting by 0.5.
    Also calculates and outputs the maximum and minimum of the original and modified vertices.

    Args:
        input_path (str): The path to the input OBJ file.
        output_path (str): The path to the output OBJ file with modified vertices.
    """
    with open(input_path, 'r') as file:
        lines = file.readlines()

    min_original = np.array([float('inf')] * 3)
    max_original = np.array([-float('inf')] * 3)
    min_modified = np.array([float('inf')] * 3)
    max_modified = np.array([-float('inf')] * 3)

    modified_lines = []
    for line in lines:
        if line.startswith('v '):  # Vertex line
            parts = line.split()
            # Convert strings to floats
            original_vertex = np.array(list(map(float, parts[1:4])))
            # Update original min and max
            min_original = np.minimum(min_original, original_vertex)
            max_original = np.maximum(max_original, original_vertex)

            # Apply transformation: vertex = vertex/2 + 0.5
            modified_vertex = original_vertex / 2 + 0.5
            # Update modified min and max
            min_modified = np.minimum(min_modified, modified_vertex)
            max_modified = np.maximum(max_modified, modified_vertex)

            modified_line = "v {:.6f} {:.6f} {:.6f}\n".format(*modified_vertex)
            modified_lines.append(modified_line)
        else:
            # Other lines are added without modification
            modified_lines.append(line)

    # Write the modified OBJ file
    with open(output_path, 'w') as file:
        file.writelines(modified_lines)

    # Print min and max values
    print(f"Original min: {min_original}, max: {max_original}")
    print(f"Modified min: {min_modified}, max: {max_modified}")
    print(f"Modified OBJ file saved as '{output_path}'")

# Usage example:
input_obj_path = 'SMPLOneDefNormalized.obj'
output_obj_path = 'SMPLOneDefNormalized2.obj'
modify_obj_vertices(input_obj_path, output_obj_path)