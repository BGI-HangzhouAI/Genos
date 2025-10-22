#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genos import create_client
from genos.plots import plot_genomic_track
from genos.utils.genomic_track_utils import *

# get the track result
api_key = None
client = create_client(api_key)
result = client.rna_coverage_track_pred("chr19", 39407000)
print(result.keys())
fig_path = "./rna_seq_coverage_track_prediction.pdf"
gff_path = "/data/work/RNASeqCovTrackPred_demo/data/gencode.v48.annotation.gff3.gz"   # replease the real path
plot_genomic_track(result['result'], gff_path, fig_path)

# get the genes name
track_result = result["result"]
chrom, start, end = track_result['position']
genes = get_genes_in_region(gff_path, chrom, start, end)
print(f"find the number of genes: {len(genes)}")
for gene in genes:
    print(f"- {gene['name']}: {gene['start']}-{gene['end']} ({gene['strand']})")

# get bigwig file for all track
fasta_path = "/data/work/RNASeqCovTrackPred_demo/data/hg38_cleaned.fa"   # replease the real path
output_path = "./bigwig_files"
result_to_bigwig(track_result, fasta_path, output_path, track_name=None)
