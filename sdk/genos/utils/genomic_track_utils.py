#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for handling genomic tracks and exporting results to BigWig format.
"""

import os
import numpy as np
import gzip
import pyBigWig


def load_gff(gff_path):
    """Load a GFF file and store genes and exons by chromosome."""
    genes = {}
    exons = {}

    try:
        open_func = gzip.open if gff_path.endswith('.gz') else open
        mode = "rt" if gff_path.endswith('.gz') else "r"
        with open_func(gff_path, mode) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                cols = line.rstrip().split("\t")
                if len(cols) < 9:
                    continue
                chrom, src, feature, start, end, score, strand, phase, attrs = cols
                start = int(start)
                end = int(end)

                # Extract gene_name / Name / gene_id
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
        print(f"Warning: Failed to load GFF file '{gff_path}': {e}")
    return genes, exons


def get_genes_in_region(gff_file, chrom, start, end):
    """
    Retrieve all genes overlapping with a given genomic region.

    This function parses a GFF (General Feature Format) file to extract gene
    annotations and identifies all genes that overlap with the specified
    genomic region (chromosome, start, end).

    Args:
        gff_file (str): Path to the GFF or GFF.gz file containing gene annotations.
        chrom (str): Chromosome name (e.g., "chr1", "chrX").
        start (int): Start position of the target genomic region (1-based).
        end (int): End position of the target genomic region (inclusive).

    Returns:
        list[dict]: A list of gene dictionaries, each containing:
            - name (str): Gene name (or "Unknown" if not found)
            - start (int): Gene start coordinate
            - end (int): Gene end coordinate
            - strand (str): "+" or "-" strand
            - source (str): Source label, e.g., "GFF"

    Example:
        >>> genes = get_genes_in_region("genes.gff3", "chr1", 100000, 200000)
        >>> for g in genes:
        ...     print(g['name'], g['start'], g['end'])
    """
    genes_in_region = []
    genes_by_chrom, _ = load_gff(gff_file)

    # Fetch genes for the specified chromosome
    genes = genes_by_chrom.get(chrom, [])

    for gene_start, gene_end, strand, gene_name in genes:
        # Check if the gene overlaps with the target region
        if not (gene_end < start or gene_start > end):
            genes_in_region.append({
                'name': gene_name if gene_name else 'Unknown',
                'start': gene_start,
                'end': gene_end,
                'strand': strand,
                'source': 'GFF'
            })

    # Sort genes by start position
    genes_in_region.sort(key=lambda x: x['start'])
    return genes_in_region


def load_chrom_sizes_from_fai(fasta_file=None):
    """
    Load chromosome sizes from a .fai file.

    Args:
        fasta_file: Path to the FASTA file. If None, reads from the FASTA_FILE environment variable.

    Returns:
        dict: A mapping {chrom: length}.
    """
    if fasta_file is None:
        fasta_file = os.getenv("FASTA_FILE")
        if not fasta_file:
            raise ValueError("Please set the FASTA_FILE environment variable.")

    fai_file = fasta_file + '.fai'

    if not os.path.exists(fai_file):
        raise FileNotFoundError(f"FAI file not found: {fai_file}")

    chrom_sizes = {}
    with open(fai_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom = parts[0]
                length = int(parts[1])
                chrom_sizes[chrom] = length

    print(f"✅ Loaded chromosome sizes for {len(chrom_sizes)} chromosomes from {fai_file}")
    return chrom_sizes


# Global cache for chromosome sizes
_CHROM_SIZES = None


def get_cached_chrom_sizes(fasta_file=None):
    """Return cached chromosome sizes, loading from .fai if not yet available."""
    global _CHROM_SIZES
    if _CHROM_SIZES is None:
        _CHROM_SIZES = load_chrom_sizes_from_fai(fasta_file)
    return _CHROM_SIZES


def result_to_bigwig(result, fasta_file, output_bw_path, track_name=None):
    """
    Convert prediction results into BigWig format.

    Args:
        result: The result dictionary returned by predictor.predict().
        output_bw_path: Output BigWig file path.
        track_name: Specific track name to export. If None, exports all tracks.
        fasta_file: Path to FASTA file. If None, reads from the FASTA_FILE environment variable.
    """
    chrom, start, end = result['position']

    # Get chromosome sizes
    CHROM_SIZES = get_cached_chrom_sizes(fasta_file)

    # Validate chromosome
    if chrom not in CHROM_SIZES:
        available_chroms = list(CHROM_SIZES.keys())
        raise ValueError(f"Chromosome '{chrom}' not found. Available: {available_chroms}")

    chrom_length = CHROM_SIZES[chrom]

    # Ensure the region is within chromosome bounds
    if end > chrom_length:
        print(f"⚠️ Warning: End position {end} exceeds chromosome {chrom} length {chrom_length}. Truncated.")
        end = chrom_length

    # Select tracks
    if track_name:
        if track_name not in result['values']:
            raise ValueError(f"Track '{track_name}' not found. Available: {list(result['values'].keys())}")
        signals = {track_name: result['values'][track_name]}
    else:
        signals = result['values']

    created_files = []

    for name, track_data in signals.items():
        # Create a separate BigWig for each track
        if track_name:
            bw_path = output_bw_path
        else:
            base_name = output_bw_path.replace('.bw', '')
            safe_name = name.replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
            bw_path = f"{base_name}_{safe_name}.bw"

        print(f"Processing track: {name}")

        signal = track_data['value']
        print("type of signal: ", type(signal), "length: ", len(signal))
        if isinstance(signal, np.ndarray):
            pass
        elif isinstance(signal, list):
            signal = np.array(signal)

        try:
            bw = pyBigWig.open(bw_path, "w")

            # Add header
            chrom_sizes_list = [(chrom, length) for chrom, length in CHROM_SIZES.items()]
            bw.addHeader(chrom_sizes_list)

            # Prepare data
            actual_length = len(signal)
            positions = np.arange(start, start + actual_length)

            # Truncate exceeding positions
            valid_mask = positions < chrom_length
            if not all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                print(f"⚠️ Warning: {invalid_count} positions exceeded chromosome length; truncated.")
                positions = positions[valid_mask]
                signal = signal[valid_mask]

            if len(positions) == 0:
                print("❌ No valid data points to write.")
                bw.close()
                continue

            # Write to BigWig
            bw.addEntries(
                [chrom] * len(positions),
                positions.tolist(),
                ends=(positions + 1).tolist(),
                values=signal.tolist()
            )

            bw.close()
            created_files.append(bw_path)
            print(f"✅ Successfully created: {bw_path}")

        except Exception as e:
            print(f"❌ Failed to create BigWig file: {e}")
            try:
                bw.close()
            except:
                pass
            if os.path.exists(bw_path):
                os.remove(bw_path)

    return created_files


def export_all_tracks_to_bigwig(result, output_dir="./bigwig_output", fasta_file=None):
    """
    Export all signal tracks in a result to separate BigWig files.

    Args:
        result: Prediction result dictionary.
        output_dir: Output directory path.
        fasta_file: Path to FASTA file. If None, reads from the FASTA_FILE environment variable.
    """
    os.makedirs(output_dir, exist_ok=True)

    chrom, start, end = result['position']
    base_filename = f"{chrom}_{start}_{end}"

    created_files = []

    print(f"🎯 Exporting all tracks to directory: {output_dir}")

    for track_name in result['values'].keys():
        safe_track_name = (track_name.replace(' ', '_')
                           .replace(':', '_')
                           .replace('(', '')
                           .replace(')', '')
                           .replace('/', '_'))

        output_path = os.path.join(output_dir, f"{base_filename}_{safe_track_name}.bw")

        files = result_to_bigwig(result, output_path, track_name=track_name, fasta_file=fasta_file)
        created_files.extend(files)

    print(f"🎉 Export complete! {len(created_files)} files created.")
    return created_files


def quick_export_single_track(result, track_name, output_path=None, fasta_file=None):
    """
    Quickly export a single track as a BigWig file.

    Args:
        result: Prediction result dictionary.
        track_name: Track name to export.
        output_path: Output file path. If None, it will be generated automatically.
        fasta_file: Path to FASTA file. If None, reads from the FASTA_FILE environment variable.
    """
    chrom, start, end = result['position']

    if output_path is None:
        safe_name = track_name.replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
        output_path = f"{chrom}_{start}_{end}_{safe_name}.bw"

    return result_to_bigwig(result, output_path, track_name=track_name, fasta_file=fasta_file)


def show_chromosome_info(fasta_file):
    """Display chromosome length information."""
    CHROM_SIZES = get_cached_chrom_sizes(fasta_file)
    print(f"Genome information (from {fasta_file}.fai):")
    print(f"Number of chromosomes: {len(CHROM_SIZES)}")
    print("\nChromosome lengths:")
    for chrom in sorted(CHROM_SIZES.keys()):
        length = CHROM_SIZES[chrom]
        print(f"  {chrom}: {length:,} bp")
