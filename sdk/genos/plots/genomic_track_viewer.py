#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import gzip
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
current_dir = os.path.dirname(os.path.abspath(__file__))
GFF_PATH = os.path.join(current_dir, "../data/gencode.v48.annotation.gff3.gz")

def _to_numpy(arr):
    """将可能是 torch tensor / numpy array / list 的 signal 转成 numpy 1d array"""
    if isinstance(arr, np.ndarray):
        return arr.squeeze()
    elif isinstance(arr, list):
        return np.array(arr).squeeze()
    else:
        pass


class GenomicTrackViewer:
    def __init__(
            self,
            gff_path=GFF_PATH,
            show_exons=True,
            exon_height=0.06,
            gene_color_plus="tab:blue",
            gene_color_minus="tab:orange",
            signal_palette=None,
            xtick_step=4000,
            dpi=100
    ):
        """
        初始化 GFF 加载器和绘图默认参数。

        Parameters:
        - gff_path: GFF/GFF3 文件路径（支持 .gz 压缩）
        - 其余参数为绘图默认值，可在 plots() 中覆盖
        """
        if gff_path is None:
            gff_path = os.getenv("GFF_FILE")
        if gff_path is None:
            raise ValueError("gff_path must be provided or set via GFF_FILE environment variable.")

        self.gff_path = gff_path
        self.show_exons = show_exons
        self.exon_height = exon_height
        self.gene_color_plus = gene_color_plus
        self.gene_color_minus = gene_color_minus
        self.signal_palette = signal_palette or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        self.xtick_step = xtick_step
        self.dpi = dpi

        # 加载 GFF 到内存：按染色体组织
        self.genes_by_chrom = {}
        self.exons_by_chrom = {}
        self._load_gff()

    def _load_gff(self):
        """一次性加载 GFF 文件，按染色体存储基因和外显子"""
        genes = {}
        exons = {}

        try:
            open_func = gzip.open if self.gff_path.endswith('.gz') else open
            mode = "rt" if self.gff_path.endswith('.gz') else "r"
            with open_func(self.gff_path, mode) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    cols = line.rstrip().split("\t")
                    if len(cols) < 9:
                        continue
                    chrom, src, feature, start, end, score, strand, phase, attrs = cols
                    start = int(start)
                    end = int(end)

                    # 提取 gene_name / Name / gene_id
                    gene_name = ""
                    for key in ["gene_name=", "Name=", "gene_id="]:
                        if key in attrs:
                            parts = [p for p in attrs.split(";") if p.startswith(key)]
                            if parts:
                                gene_name = parts[0].split("=", 1)[1].strip('"').strip()
                                break

                    if feature == "gene":
                        genes.setdefault(chrom, []).append((start, end, strand, gene_name))
                    elif feature == "exon":
                        exons.setdefault(chrom, []).append((start, end, strand, gene_name))
        except Exception as e:
            print(f"Warning: Failed to load GFF file '{self.gff_path}': {e}")
            # 即使失败也允许后续绘图（显示“No genes”）

        self.genes_by_chrom = genes
        self.exons_by_chrom = exons

    def plot(
            self,
            data,
            show_exons=None,
            exon_height=None,
            gene_color_plus=None,
            gene_color_minus=None,
            signal_palette=None,
            xtick_step=None,
            dpi=None
    ):
        """
        可视化基因组信号与基因结构。

        Parameters 与原函数一致，但 gff_path 已在初始化时指定。
        所有参数均可在此覆盖类默认值。
        """
        # 使用传入参数或回退到类属性
        show_exons = show_exons if show_exons is not None else self.show_exons
        exon_height = exon_height if exon_height is not None else self.exon_height
        gene_color_plus = gene_color_plus if gene_color_plus is not None else self.gene_color_plus
        gene_color_minus = gene_color_minus if gene_color_minus is not None else self.gene_color_minus
        signal_palette = signal_palette if signal_palette is not None else self.signal_palette
        xtick_step = xtick_step if xtick_step is not None else self.xtick_step
        dpi = dpi if dpi is not None else self.dpi

        labels = data["values"]
        chrom, start, end = data["position"]

        # 确定显示区间
        first_track_name = list(labels.keys())[0]
        first_signal = _to_numpy(labels[first_track_name]['value'])
        actual_data_length = len(first_signal)
        start_display = int(start)
        end_display = int(start + actual_data_length)
        display_positions = np.arange(start_display, end_display)
        track_names = list(labels.keys())

        # 创建图形
        n_tracks = len(track_names)
        fig, axes = plt.subplots(
            n_tracks + 1, 1,
            figsize=(12, 3 * (n_tracks + 1.0)),
            sharex=True,
            gridspec_kw={'height_ratios': [1.0] + [1.0] * n_tracks},
            dpi=dpi
        )
        if n_tracks + 1 == 1:
            axes = [axes]
        ax_gene = axes[0]

        # 获取当前染色体的基因/外显子
        genes = self.genes_by_chrom.get(chrom, [])
        exons = self.exons_by_chrom.get(chrom, []) if show_exons else []

        # 过滤到显示区间
        filtered_genes = [
            (gs, ge, strand, name) for gs, ge, strand, name in genes
            if not (ge < start_display or gs >= end_display)
        ]
        filtered_exons = [
            (es, ee, strand, name) for es, ee, strand, name in exons
            if not (ee < start_display or es >= end_display)
        ]

        # 绘制基因轨道
        if filtered_genes:
            genes_df = pd.DataFrame(filtered_genes, columns=["start", "end", "strand", "name"])
            genes_sorted = genes_df.sort_values("start").reset_index(drop=True)

            # 基因布局算法
            level_ends = []
            level_height_base = 0.3
            level_gap = 0.25
            max_levels = 8
            gene_levels = []

            for _, row in genes_sorted.iterrows():
                gs, ge = int(row["start"]), int(row["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest_end_idx = np.argmin(level_ends)
                    level_ends[earliest_end_idx] = ge
                    gene_levels.append(earliest_end_idx)

            # 绘制基因
            for (idx, row), lvl in zip(genes_sorted.iterrows(), gene_levels):
                gs, ge, strand, name = int(row["start"]), int(row["end"]), row["strand"], row["name"]
                y = level_height_base + lvl * level_gap
                color = gene_color_plus if strand == "+" else gene_color_minus

                # 主干线
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.5, zorder=2, solid_capstyle='round')

                # 箭头
                gene_length = ge - gs
                arrow_length = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_length, y, arrow_length, 0,
                                  head_width=0.04, head_length=arrow_length * 0.3,
                                  fc=color, ec=color, linewidth=0,
                                  length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_length, y, -arrow_length, 0,
                                  head_width=0.04, head_length=arrow_length * 0.3,
                                  fc=color, ec=color, linewidth=0,
                                  length_includes_head=True, zorder=3)

                # 外显子
                if show_exons:
                    gene_exons = [e for e in filtered_exons if e[3] == name]
                    for exon in gene_exons:
                        es, ee = int(exon[0]), int(exon[1])
                        es_d = max(es, start_display)
                        ee_d = min(ee, end_display)
                        if ee_d > es_d and (ee_d - es_d) > 50:
                            rect = mpatches.Rectangle(
                                (es_d, y - exon_height / 2),
                                ee_d - es_d, exon_height,
                                facecolor=color, alpha=0.9, zorder=3,
                                edgecolor='white', linewidth=0.5
                            )
                            ax_gene.add_patch(rect)

                # 基因名（简单居中，避免复杂重叠逻辑）
                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                ax_gene.text(text_x, y + exon_height + 0.03,
                             name if name else "Unknown",
                             ha="center", va="bottom", fontsize=9, zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                       alpha=0.8, edgecolor='none'))

            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.2
            ax_gene.set_ylim(0, gene_track_height)

            # 图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=gene_color_plus, lw=2, marker='>', markersize=8,
                       label='Forward strand (+)'),
                Line2D([0], [0], color=gene_color_minus, lw=2, marker='<', markersize=8,
                       label='Reverse strand (-)')
            ]
            ax_gene.legend(handles=legend_elements, loc='upper right', fontsize=10,
                           framealpha=0.9, fancybox=True)
        else:
            ax_gene.text(0.5, 0.5, "No genes in this region",
                         ha="center", va="center", transform=ax_gene.transAxes,
                         fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)

        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')
        ax_gene.set_title(f"{chrom}:{start_display:,}-{end_display:,}  (Length: {actual_data_length:,} bp)",
                          loc="left", fontsize=10, pad=15)

        # 绘制信号轨道
        for i, name in enumerate(track_names):
            ax = axes[i + 1]
            signal = _to_numpy(labels[name]['value'])
            color = signal_palette[i % len(signal_palette)]

            ax.plot(display_positions, signal, color=color, linewidth=1.5, alpha=0.8)
            ax.fill_between(display_positions, 0, signal, alpha=0.25, color=color)

            y_max = max(signal.max() * 1.15, 0.1) if len(signal) > 0 else 1
            ax.set_ylim(0, y_max)
            ax.set_yticks(np.linspace(0, y_max, 5))
            ax.tick_params(axis='y', labelsize=9)
            ax.set_ylabel("Signal\nIntensity", fontsize=11, rotation=0, ha='right', va='center')
            ax.set_title(name, fontsize=10, pad=10)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # X轴设置
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if xtick_step and actual_data_length > xtick_step:
                xticks = np.arange(start_display, end_display, xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=10)

        def format_genomic_position(x, pos):
            return f"{int(x):,}"

        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_position))

        axes[-1].set_xlabel(f"Genomic Position ({chrom})", fontsize=12, labelpad=10)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.subplots_adjust(hspace=0.15)
        plt.show()

        return fig, axes


def plot_genomic_track(data, gff_path=GFF_PATH, save_path=None):
    viewer = GenomicTrackViewer(gff_path=gff_path)
    fig, _ = viewer.plot(data)
    if save_path is not None:
        fig.savefig(save_path, format='pdf')
    return fig