import os
import os.path as osp
import shutil
import json
import csv
from PIL import Image
import numpy as np


def reducing_waypoints_list(waypoints_list, key_list):
    reduced_waypoints_list = []
    for waypoint in waypoints_list:
        reduced_waypoint = dict((k, waypoint[k]) for k in key_list)
        reduced_waypoints_list.append(reduced_waypoint)
    return reduced_waypoints_list


def writing_waypoints_to_csv(export_path, waypoints_list):
    keys = waypoints_list[0].keys()
    with open(export_path, "w", newline="") as output_file:
        output_file.write("# The map name is '$mapname'\n")
        output_file.write("# The waypoints are from the optimal race line.\n")
        output_file.write("# ")
        dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
        dict_writer.writeheader()
        dict_writer.writerows(waypoints_list)


def convert_wayponts_list_to_csv(
    current_map_name, waypoints_list, trajectory_name, export_dir_path
):
    export_raceline_path = (
        export_dir_path + "/" + f"{current_map_name}_{trajectory_name}.csv"
    )
    export_extended_raceline_path = (
        export_dir_path + "/" + f"{current_map_name}_extended_{trajectory_name}.csv"
    )

    key_list = ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"]

    writing_waypoints_to_csv(
        export_raceline_path, reducing_waypoints_list(waypoints_list, key_list)
    )
    writing_waypoints_to_csv(export_extended_raceline_path, waypoints_list)


def convert_map_to_greyscale(path):
    img = Image.open(path)
    grayscale_img = img.convert("L")
    grayscale_img.save(path)
    map_img = np.array(Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)).astype(
        np.float64
    )
    np.savetxt(path + ".csv", map_img, delimiter=",")


def create_new_dir(dir):
    if not osp.exists(dir):
        os.mkdir(dir)


def main():
    map_convertor_dir_path = os.path.dirname(os.path.realpath(__file__))
    map_import_dir_path = osp.join(map_convertor_dir_path, "map_imported")
    map_export_dir_path = osp.join(map_convertor_dir_path, "map_exported")
    create_new_dir(map_export_dir_path)

    maps_to_import_list = [
        name
        for name in os.listdir(map_import_dir_path)
        if os.path.isdir(os.path.join(map_import_dir_path, name))
    ]

    for current_map_name in maps_to_import_list:
        current_map_import_dir_path = osp.join(map_import_dir_path, current_map_name)
        current_map_export_dir_path = osp.join(map_export_dir_path, current_map_name)

        create_new_dir(current_map_export_dir_path)

        convert_map_to_greyscale(
            osp.join(current_map_import_dir_path, f"{current_map_name}.png")
        )

        files_to_be_copied = [f"{current_map_name}.png", f"{current_map_name}.yaml"]
        for file in files_to_be_copied:
            shutil.copy(
                osp.join(current_map_import_dir_path, file),
                osp.join(current_map_export_dir_path, file),
            )

        global_waypoints_dir_path = osp.join(
            current_map_import_dir_path, "global_waypoints.json"
        )
        global_waypoints_data = json.load(open(global_waypoints_dir_path))
        selected_trajectory = "global_traj_wpnts_iqp"
        waypoints_to_be_saved_list = global_waypoints_data[selected_trajectory]["wpnts"]
        trajectory_name = "raceline"
        convert_wayponts_list_to_csv(
            current_map_name,
            waypoints_to_be_saved_list,
            trajectory_name,
            current_map_export_dir_path,
        )

        selected_trajectory = "centerline_waypoints"
        waypoints_to_be_saved_list = global_waypoints_data[selected_trajectory]["wpnts"]
        trajectory_name = "centerline"
        convert_wayponts_list_to_csv(
            current_map_name,
            waypoints_to_be_saved_list,
            trajectory_name,
            current_map_export_dir_path,
        )


if __name__ == "__main__":
    main()
