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
    

    def generate_bar_chart(self, tuples, fig_file, title, xlabel, ylabel):
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
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


        plt.tight_layout()
        plt.savefig(fig_file)
        plt.close()


    def _calculate_legend(self, generate_legend, legend_x, legend_y, percent_of_event_above, percent_of_event_below,
                          percentage_threshold, reverse_order, x, y, bar_labels):
        ticks = range(len(x))
        colors = ['red'] * len(ticks)
        if percent_of_event_below > 0 and reverse_order:
            colors[-1] = 'blue'
        f, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.bar(ticks, y, color=colors, label=range(len(ticks)))
        if bar_labels:
            for i in range(len(x)):
                plt.text(i, round(y[i],1)+0.2, round(y[i],1), ha='center', fontsize=9)
        #plt.plot(ticks, y, color='red')
        blue_patch = mpatches.Patch(color='blue',
                                    label=f'{percentage_threshold}% of {legend_y} produced by'
                                          f' {percent_of_event_below}% of {legend_x if legend_x else ""}')
        red_patch = mpatches.Patch(color='red',
                                   label=f'{100 - percentage_threshold}% of {legend_y} produced by'
                                         f' {percent_of_event_above}% of {legend_x} if legend_x else ""')
        if percent_of_event_below > 0 and generate_legend:
            plt.legend(handles=[blue_patch, red_patch], fontsize=self.default_legend_font_size,
                       loc='upper left')

    def _plot_the_chart(self, fig_file, legend_x, legend_y, sorted_tuples, title,
                        suptitle,
                        total_y_values, x, xlabel,
                        y_in_percent, ylabel):
        plt.xticks(range(len(x)), x, rotation=self.x_ticks_rotation,
                   fontsize=self.default_ticks_font_size)
        plt.yticks(fontsize=self.default_ticks_font_size)
        plt.xlabel(xlabel, fontsize=self.default_axis_labels_font_size)
        plt.ylabel(ylabel, fontsize=self.default_axis_labels_font_size)

        legend_x_title = f', {legend_x}={len(sorted_tuples)}' if legend_x else ""
        plt.suptitle(title+"\n"+suptitle, fontsize=14)

        plt.title(f"({legend_y}={int(total_y_values)}{legend_x_title})",
                      fontsize=10)
        plt.savefig(fig_file, format=self.fig_formet, bbox_inches="tight")

    def _calculate_threshold_split(self, increasing_sorted_tuples, percentage_threshold, reverse_order, total_y_values):
        grouping_data = self._group_below_threshold_x_values(increasing_sorted_tuples, percentage_threshold,
                                                             total_y_values, reverse_order)
        other = grouping_data["other_label"]
        threshold = grouping_data["threshold"]
        percent_of_event_above = grouping_data["p_event_above_th"]
        percent_of_event_below = grouping_data["p_events_below_th"]
        return other, percent_of_event_above, percent_of_event_below, threshold

    def _calculate_y_labels(self, other, increasing_sorted_tuples, threshold,
                            total_y_values, y_in_percent, reverse_order, other_index,
                            accumulated=False):
        if total_y_values == 0:
            raise ValueError("Total y values is 0")
        y = [t[1] for t in increasing_sorted_tuples if t[1] > threshold]
        if y_in_percent:
            y = [(v/total_y_values)*100 for v in  y]
            other_percent =  (other[0], (other[1]/total_y_values)*100)
            y.insert(other_index, other_percent[1])
        else:
            y.insert(other_index,other[1])
        if accumulated:
            y = [round(sum(y[:index+1]),4) for index, _ in enumerate(y)]
        return y

    def _skip_labels(self, x):
        if len(x) < 15:
            return
        for index, _ in enumerate(x):
            if index % 3 != 0:
                x[index] = ''

    def generate_line_chart(self, fig_file, tuples, xlabel, ylabel, title,
                            watermark=False,
                            legend=False, color="g", threshold=500):
        x = [x[0] for x in tuples]
        y = [x[1] for x in tuples]
        plt.figure(dpi=120)
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Search Latency')
        plt.xticks(rotation=45, fontsize=self.default_ticks_font_size-5)
        plt.yticks(fontsize=self.default_ticks_font_size-5)
        plt.xlabel(xlabel, fontsize=self.default_axis_labels_font_size)
        plt.ylabel(ylabel, fontsize=self.default_axis_labels_font_size)
        plt.title(title, fontsize=self.default_title_font_size, y=1)
        if legend:
            plt.legend()
        if watermark:
            raise NotImplementedError('watermark not implemented')

        plt.savefig(fig_file, format=self.fig_formet, bbox_inches="tight")




    def _group_below_threshold_x_values(self, increasing_sorted_order, threshold_percent_of_total,
                                        total_y_value, reverse_order):

        n_tuples = len(increasing_sorted_order)
        commutative_total = 0
        threshold = 0
        below_threshold_percent= 0
        above_threshold_percent = 100
        for index, tuple in enumerate(increasing_sorted_order):
            if commutative_total <= total_y_value * (threshold_percent_of_total/100):
                commutative_total += tuple[1]
            else:
                test=sum([v for x, v in increasing_sorted_order[index:]])
                threshold = increasing_sorted_order[index-1][1]
                below_threshold_percent = (index-1) / n_tuples
                above_threshold_percent = (n_tuples - index-1) / n_tuples
                below_threshold_percent = round(below_threshold_percent * 100, 1)
                above_threshold_percent= round(above_threshold_percent * 100, 1)
                assert test == total_y_value - commutative_total
                break

        other = self._get_other_group_label(increasing_sorted_order, threshold)
        return {"other_label":other,
                "p_event_above_th":above_threshold_percent,
                "p_events_below_th":below_threshold_percent,
                "threshold": threshold}

    def _get_other_group_label(self, sorted_tuples, threshold):
        count_other = sum([x[1] for x in sorted_tuples if x[1] <= threshold])
        other = ('all others', count_other)
        return other

    def _set_custom_labels(self, x):
        return ['prod-stage', 'glue-info', 'glue-error']
    
    def generate_scatter_chart(self, file, x, y, title="Scatter Plot", 
                               xlabel="X-axis", ylabel="Y-axis", subtitle=None, 
                              color="blue", size=5):

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=color, s=size, alpha=0.6, edgecolors='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if subtitle:
            plt.suptitle(subtitle, fontsize=10, fontweight='light')
        #plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(file, dpi=300, bbox_inches='tight')
        #plt.show()

    def generate_histogram(self, data, bins=10, title="Histogram", 
                               xlabel="Values", ylabel="Frequency", 
                               color="blue", filename="histogram.png"):
        
            plt.figure(figsize=(8, 6))
            plt.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black', 
                     weights=np.ones(len(data)) / len(data) * 100)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            # Get the bin edges
            ticks = np.arange(1, bins)  # Sequence of integers from 1 to 10
            plt.xticks(ticks)
            #plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            #plt.show()

    def generate_histogram_2d(self, file_name, data, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency",
                              color="blue", subtitle=None, organization='not_set'):
        np_data = np.array(data)
        total_elems = np.sum(np_data)
        np_data = np.divide(np_data, total_elems)
        DF = pd.DataFrame(np_data)
        DF.to_csv("/tmp/data1.csv")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title, fontsize=self.default_title_font_size, y=1.05)
        #fig.tight_layout()
        num_rows, num_cols = np_data.shape
        ax.set_xlim([0, num_cols+0.25])
        ax.set_xticks(np.arange(0.5, num_cols,1),labels=range(1,num_cols+1) ,fontsize=10)
        ax.set_yticks(list(np.arange(2,num_rows+2,2))+[num_rows+1], labels=list(range(2,num_rows,2))+['26+',''],fontsize=10)

        zones = self._color_heath_zones(ax, num_cols, num_rows)
        zone_values = defaultdict(lambda: 0)
        self._calculate_health_zone_values(np_data, num_cols, num_rows, zone_values, zones)
        plt.xlabel(xlabel, fontsize=self.default_axis_labels_font_size-2)
        plt.ylabel(ylabel, fontsize=self.default_axis_labels_font_size-2)
        self._add_values_to_individual_cells(ax, np_data, num_cols, num_rows)
        self._add_legend(zone_values, organization)
        plt.text(1.35, 1.9, '    ',
                  color='black', fontsize=8, backgroundcolor='white', linespacing=1.15 ,
                  bbox=dict(facecolor='none', edgecolor='blue', lw=1, boxstyle='square'))

        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    def _add_legend(self, zone_values, organization):
        color_handles = []
        for color in [('green', '`Healthy`'), ('yellow', '`Unhealthy`'), ('red', '`Dead`')]:
            patch_label = f"{zone_values[color[0]] * 100:.1f}% of {organization} methods are {color[1]}"
            color_handles.append(mpatches.Patch(color=color[0], label=patch_label))
        plt.legend(handles=color_handles, ncols=1, #bbox_to_anchor=(0.5, 1.03),
                   loc='lower right', fontsize=self.default_legend_font_size-7)
    def _add_values_to_individual_cells(self, ax, np_data, num_cols, num_rows):
        for i in range(num_rows):
            for j in range(num_cols):
                value = np_data[i, j] * 100.0
                if value > 0.05:
                    ax.text(j+0.5, i+1, f'{value:.1f}', fontsize=8,
                            ha="center", va="center", color="b")

    def _calculate_health_zone_values(self, np_data, num_cols, num_rows, zone_values, zones):
        for color, color_zones in zones.items():
            #if color != 'yellow':
            #    continue
            # sum=0
            for zone in color_zones:
                for x in range(num_cols):
                    for y in range(num_rows):
                                (x_min, y_min), (x_max, y_max) = ((zone[0], zone[1]),
                                                                  (zone[0] + zone[2], zone[1] + zone[3]))
                                if x_min <= x < x_max and y_min <= y+1 <= y_max:
                                    value = np_data[y,x]
                                    zone_values[color] += value
                                    # if x == 3:
                                    #     sum += value
                                    #     print(f'({x}, {y}) in {color}, {value}, {sum}')
        return
    def _color_heath_zones(self, ax, num_cols, num_rows):
        zones = {"green": [(-1, -1, num_cols + 1.25, 12.5),
                           (2, 11.5, num_cols + .5, 7),
                           (3, 18.5, num_cols + .5, 7)],
                  "yellow": [(0, 11.5, 2, 7),
                             (2, 18.5, 1, 7),
                            ],
                 "red": [(0, 18.5, 2, num_rows - 12 - 7), (0, num_rows-0.5, num_cols + 1.5, 1.5)],
                 "gray": [(x+1,x,num_cols +1,1.5) for x in range(num_cols)],
                 }
        for color, color_zones in zones.items():
            for zone in color_zones:
                ax.add_patch(mpatches.Rectangle((zone[0], zone[1]), zone[2], zone[3], color=color))
        return zones
