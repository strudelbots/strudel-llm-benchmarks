from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

class ChartGenerator():

    def __init__(self, default_ticks_font_size=12, default_axis_labels_font_size=18,
                 x_ticks_rotation=45, add_annotations=True):
        self.fig_width = 10
        self.fig_height = 7
        self.default_ticks_font_size = default_ticks_font_size
        self.default_axis_labels_font_size = default_axis_labels_font_size
        self.default_legend_font_size = 16
        self.default_title_font_size = 16
        self.default_subtitle_font_size = 18
        self.default_annotation_font_size = 10
        self.x_ticks_rotation = x_ticks_rotation
        self.add_annotations = add_annotations
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

        ax.set_title(title, fontsize=self.default_title_font_size, weight='bold')
        ax.set_xlabel(xlabel, fontsize=self.default_axis_labels_font_size)
        ax.set_ylabel(ylabel, fontsize=self.default_axis_labels_font_size)
        ax.tick_params(axis='x', rotation=self.x_ticks_rotation, labelsize=self.default_ticks_font_size)
        if log_scale:
            ax.set_yscale('log')
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            annotation_str = f'{height:.{annotation}f}'
            if self.add_annotations:
                ax.annotate(annotation_str,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=self.default_annotation_font_size)


        plt.tight_layout()
        print(f"**  Saving chart to {fig_file}")
        plt.savefig(fig_file)
        plt.close()

    def create_scatter_plot(self, x, y, labels, filename, xlabel, ylabel, title):
        """
        Creates and saves a scatter plot with labels for each data point.

        Parameters:
        - x: list of x-coordinates
        - y: list of y-coordinates
        - labels: list of labels for each point
        - filename: output file name (default: 'scatter_plot.png')
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue')

        # Add labels to each point
        for i in range(len(x)):
            plt.text(x[i] + 0.1, y[i] + 0, labels[i], fontsize=8)

        # Add title and axis labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Save the figure to a file
        plt.savefig(filename, dpi=300)

    def create_heat_map(self, similarity_df, fig_file, title, mask_lower_triangle=True):    # Create a heatmap
        plt.figure(figsize=(12, 12))
        if mask_lower_triangle:
            mask = np.tril(np.ones(similarity_df.shape), k=-1).astype(bool)
        else:
            mask = None
        ax = sns.heatmap(similarity_df, annot=True, cmap='jet', fmt=".2f", mask=mask,
                    vmin=0, vmax=1)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=self.default_ticks_font_size+7, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=self.default_ticks_font_size+7, rotation=0)
        plt.title(title, fontsize=20)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95]) 
        plt.savefig(fig_file)
        plt.close()