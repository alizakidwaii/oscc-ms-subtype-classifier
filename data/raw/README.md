# Raw Data Sources

This folder is intentionally not version-controlled for the large files (see `.gitignore`) — they're freely re-downloadable from their original public sources. This keeps the repository small while remaining fully reproducible. The already-processed dataset (`data/oscc_gene_expression.csv`) is committed to the repo, so `train`/`evaluate`/`predict` work immediately after cloning without these raw files at all.

## 1. Gene expression matrix — `TCGA_HNSC_HiSeqV2.gz`

- **Source:** UCSC Xena Browser, TCGA Head and Neck Cancer (HNSC) cohort
- **URL:** https://xenabrowser.net/datapages/ → search "Head and Neck" → **TCGA Head and Neck Cancer (HNSC)** → **gene expression RNAseq** section → **"IlluminaHiSeq (n=566) TCGA Hub"** → Download
- Save as `data/raw/TCGA_HNSC_HiSeqV2.gz`

## 2. Molecular subtype labels — `TCGA_HNSC_subtypes_TableS7.2.xlsx`

- **Source:** Supplementary Data of the TCGA HNSC study, Nature 517, 576-582 (2015). DOI: 10.1038/nature14129
- **URL:** https://www.nature.com/articles/nature14129 → "Supplementary information" → **"Supplementary Data (download ZIP)"**
- Inside the ZIP, use file **`7.2.xlsx`**, sheet **`Platform_Class_Labels`**
- Save as `data/raw/TCGA_HNSC_subtypes_TableS7.2.xlsx`

## Rebuilding the dataset

Once both files are in place, run:
```
python main.py build-dataset
```