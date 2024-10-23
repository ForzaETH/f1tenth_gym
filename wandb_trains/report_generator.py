#!/usr/bin/env python3
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cairosvg
from PyPDF2 import PdfMerger
import pickle
import seaborn as sns
from plotting_utils import prettify_label, base_plot, load_config


class ReportGenerator:
    def __init__(self, recorder_config : dict, plot_config, trajectory_name="Min Curvature Path", save_dir: str =None):
        self.recorder_config = recorder_config
        self.plot_config = plot_config
        if save_dir is None:    
            self.save_root_dir = (
                os.path.expanduser("~") + self.recorder_config["save_root_dir"]
            )
        else:
            self.save_root_dir = save_dir
        self.car_type = self.recorder_config["last_car_type"]
        self.time_stamp = self.recorder_config["last_time_stamp"]
        save_dir = osp.normpath(
            self.save_root_dir
            + os.sep
            + f"{self.car_type}_recordings"
            + os.sep
            + self.time_stamp
        )
        self.save_dir = os.path.expanduser(save_dir)
        car_info_path = osp.join(self.save_dir, f"car_raw_info.csv")
        
        if osp.exists(car_info_path):
            self.car_raw_info_df = pd.read_csv(car_info_path, delim_whitespace=True)
        else:
            raise FileNotFoundError(f"Car info file not found: {car_info_path}")

        self.load_info_unit_mapping()
        self.trajectory_name = trajectory_name
        self.save_dir_plots = osp.join(self.save_dir, "plots")
        os.makedirs(self.save_dir_plots, exist_ok=True)

    def load_info_unit_mapping(self):
        save_path_info_unit_mapping = osp.join(self.save_dir, "info_unit_mapping.pkl")
        if osp.exists(save_path_info_unit_mapping):
            with open(save_path_info_unit_mapping, "rb") as file:
                self.info_unit_mapping = pickle.load(file)
        else:
            raise FileNotFoundError(
                f"Info unit mapping file not found: {save_path_info_unit_mapping}"
            )

    def load_track_and_traj(self):
        save_path_track = osp.join(self.save_dir, "track_bound.npy")
        if osp.exists(save_path_track):
            self.track_bound_coords = np.load(rf"{save_path_track}")
        else:
            raise FileNotFoundError(f"Track bound file not found: {save_path_track}")

        save_path_traj = osp.join(self.save_dir, "traj.npy")
        if osp.exists(save_path_traj):
            self.global_waypoints_coords = np.load(rf"{save_path_traj}")
        else:
            raise FileNotFoundError(f"Trajectory file not found: {save_path_traj}")

    def run(self):
        self.load_track_and_traj()
        for info in list(self.car_raw_info_df)[3:]:
            self.plot_info_diagram(info)

        if self.recorder_config["create_pdf_report"]:
            self.generate_report_svg()

    def generate_report_svg(self):
        print("Generating PDF report...")

        temp_pdf_directory = osp.join(self.save_dir, "tmp")
        os.makedirs(temp_pdf_directory, exist_ok=True)

        merger = PdfMerger()

        for subdir, _, files in os.walk(self.save_dir):
            for filename in files:
                if filename.endswith(".svg"):
                    svg_filepath = os.path.join(subdir, filename)
                    temp_pdf_filepath = os.path.join(
                        temp_pdf_directory, filename.replace(".svg", ".pdf")
                    )
                    cairosvg.svg2pdf(url=svg_filepath, write_to=temp_pdf_filepath)
                    merger.append(temp_pdf_filepath)

        output_pdf_path = osp.join(self.save_dir, f"output_{self.time_stamp}.pdf")
        merger.write(output_pdf_path)
        merger.close()

        print("PDF report is ready!")

    def plot_info_diagram(self, info_name):
        sns.set(style="whitegrid")
        df = self.car_raw_info_df
        max_val = df[info_name].max()
        min_val = df[info_name].min() if info_name != "speed" else 0

        unit = self.info_unit_mapping.get(info_name, "")
        unit_label = rf"${unit}$" if unit else ""
        recording_duration = round(df["time"].max() - df["time"].min(), 3)
        test_env_name = "**Gym Environment**"

        # Create the base plot layout
        fig, ax = base_plot(self.track_bound_coords, self.global_waypoints_coords, self.plot_config)

        cmap = plt.get_cmap("PuRd")
        norm = plt.Normalize(min_val, max_val)

        if info_name == "speed":
            norm = plt.Normalize(*self.plot_config["norm_speed"])

        # Plot the car data points
        points = ax.scatter(
            df["pos_x"],
            df["pos_y"],
            c=df[info_name],
            cmap=cmap,
            norm=norm,
            s=self.plot_config["car_info_size"],
            alpha=0.6,
            label=f"{prettify_label(info_name)} in {unit_label}",
        )
        cbar = fig.colorbar(
            points,
            ax=ax,
            orientation=self.plot_config["cbar_orientation"],
            pad=self.plot_config["cbar_pad"],
        )
        cbar.set_label(
            f"{prettify_label(info_name)} {unit_label}",
            fontsize=self.plot_config["axis_fontsize"],
            weight=self.plot_config["axis_fontweight"],
        )

        ax.legend(
            [self.trajectory_name, "Driven Path", "Track Boundary Points"],
            loc=self.plot_config["legend_loc"],
            borderaxespad=0.2,
        )

        # Set the plot title and adjust layout
        title = f"{test_env_name}\nRecording Of The {prettify_label(info_name)} For {recording_duration} Seconds"
        ax.set_title(
            title,
            fontsize=self.plot_config["title_fontsize"],
            weight=self.plot_config["title_fontweight"],
            pad=30,
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        output_format = self.plot_config.get("output_format", "svg")
        save_path = osp.join(
            self.save_dir_plots,
            f"{info_name}_plot.{output_format}",
        )
        save_path = osp.normpath(save_path)
        fig.savefig(save_path, format=output_format, dpi=1200, bbox_inches="tight")
        print(f"The plot recording the {info_name} is saved to {save_path}")

        plt.close(fig)


if __name__ == "__main__":
    report_generator = ReportGenerator(load_config("recorder_config"), load_config("plot_config"))
    report_generator.run()
