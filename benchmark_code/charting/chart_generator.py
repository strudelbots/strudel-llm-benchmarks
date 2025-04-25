from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


class ChartGenerator():

    def __init__(self, default_ticks_font_size=16, default_axis_labels_font_size=16,
                 x_ticks_rotation=25):
        self.fig_width = 10
        self.fig_height = 7
        self.default_ticks_font_size = default_ticks_font_size
        self.default_axis_labels_font_size = default_axis_labels_font_size
        self.default_legend_font_size = 16
        self.default_title_font_size = 16
        self.default_suptitle_font_size = 18
        self.x_ticks_rotation = x_ticks_rotation
        self.fig_formet = "png"
    

    def generate_bar_chart(self, tuples, fig_file, title, xlabel, ylabel,
                           log_scale=False, annotation = '3'):
        """
        Generates a bar chart and saves it to a file.

        Parameters:
        - tuples: List of (label, value) tuples
        - fig_file: Output file path for the chart
        - title: Main title of the chart
        - xlabel: Label for the X-axis
        - ylabel: Label for the Y-axis
        - subtitle: Subtitle of the chart
        """
        labels, values = zip(*tuples)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color='skyblue')

        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel(xlabel, fontsize=self.default_axis_labels_font_size)
        ax.set_ylabel(ylabel, fontsize=self.default_axis_labels_font_size)
        ax.tick_params(axis='x', rotation=45)
        if log_scale:
            ax.set_yscale('log')
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            annotation_str = f'{height:.{annotation}f}'
            ax.annotate(annotation_str,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.default_ticks_font_size-3)


        plt.tight_layout()
        plt.savefig(fig_file)
        plt.close()

