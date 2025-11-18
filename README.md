## ImmunoGeNN
ImmunoGeNN accepts input protein sequences and predicts peptide MHC-II immunogenicity risk scores (pIRS) based on allele frequencies in the global population. It predicts risk scores at up to over 300.000 times the rate of NetMHCIIpan-4.3, while sharing a >95% Spearman R correlation on the NetMHCIIpan-4.3 test set and ~45% with experimentally presented peptides (MAPPs - see [pre-print](https://openreview.net/forum?id=kOJQm9YXnB) by Høie et al 2025).

Example FASTA input:
```
>LYZL4_MOUSE Lysozyme-like protein 4
MQLYLVLLLISYLLTPIGASILGRCTVAKMLYDGGLNYFEGYSLENWVCLAYFESKFNPS
AVYEDPQDGSTGFGLFQIRDNEWCGHGKNLCSVSCTALLNPNLKDTIQCAKKIVKGKHGM
GAWPIWSKNCQLSDVLDRWLDGCDL
```

### Paper and code
For more details please see the [pre-print](https://openreview.net/forum?id=kOJQm9YXnB) by Høie et al 2025, "ImmunoGeNN: Accelerating Early Immunogenicity Assessment for Generative Design of Biologics", presented at the EurIPS 2025 Workshop on SIMBIOCHEM.

- [Pre-print](https://openreview.net/forum?id=kOJQm9YXnB)
- [GitHub repository](https://github.com/novonordisk-research/ImmunoGeNN)
- [Biolib web-server](https://biolib.com/DTU/ImmunoGeNN)
- [DTU Health web-server](https://services.healthtech.dtu.dk/services/ImmunoGeNN/)

----

### Input format
- Input FASTA file of proteins sequences (minimum sequence length of 15 residues)
- Human reference - Toggle on to set peptide IRS scores to zero if (binding core) is observed in the human reference. Additional reference sequences may be added. Default on.
- Deimmunize first sequence - Toggle on to iteratively screen all possible single amino acid variants for deimmunizing effects on the first input sequence. Default off.
- Human reference filtering uses the top identified binding core and searches for matches in the human proteome.

### Output format

ImmunoGeNN predicts per-peptide IRS scores ("pIRS") for the Global population, DRB1 gene class. Sequence pIRS_sum scores are calculated by summing across all peptide pIRS scores in the given sequence.

**pIRS score interpretation:** We suggest using a pIRS rank threshold of ~83% to identify immunogenic peptides, as described in the paper. Higher scores indicate higher global population (MHC-II presentation) immunogenicity risk, with an estimated experimental Spearman R MAPPs correlation of ~0.45.

### Example output

pIRS.csv - CSV file containing per-peptide pIRS scores
```csv
id,peptide_pos,gene_class,peptide_seq,core_pos,core_seq,pIRS,pIRS_rank,in_reference,core_0,core_1,core_2,core_3,core_4,core_5,core_6
design1,1,DRB1,EVQLLESGGEVKKPG,3,LLESGGEVK,0.03498,44.949,,0.00,0.00,13.96,67.79,18.25,0.00,0.00
design1,2,DRB1,VQLLESGGEVKKPGA,3,LESGGEVKK,0.03122,36.630,,11.49,0.00,31.90,56.62,0.00,0.00,0.00
design1,3,DRB1,QLLESGGEVKKPGAS,2,LESGGEVKK,0.02773,20.586,,0.00,14.51,71.27,0.00,0.00,0.00,14.22
```
- id: Sequence identifier from input FASTA
- peptide_pos: Position of peptide in sequence (1-indexed)
- gene_class: Always MHC-II DRB1 gene class
- peptide_seq: Peptide sequence
- core_pos: Position of dominant DRB1 binding core in peptide (0-indexed)
- core_seq: DRB1 binding core sequence
- pIRS: Predicted immunogenicity risk score. Higher is more immunogenic
- pIRS_rank: Predicted rank in NetMHCIIpan-4.3 training set (see paper). Higher is more immunogenic above a threshold of ~83%.
- in_reference: Set to True if the 9-mer core sequence is found in human reference proteome
- core_0 to core_6: Predicted peptide IRS for all 7 binding cores, corresponding to model confidence. The top binding core is picked based on the highest pIRS score. See paper for more details.

scores.csv - CSV file containing per-sequence pIRS scores (summed across all peptides)
```csv
id,population,DRB1_pIRS_sum
design1,Global,5.16455
design2,Global,5.17534
design3,Global,5.13089
```
- id: Sequence identifier from input FASTA
- population: Population for which pIRS is calculated
- DRB1_pIRS_sum: Sum of pIRS scores across all peptides in the sequence

----

### Download and run locally

Installation:
```
git clone https://github.com/novonordisk-research/ImmunoGeNN
cd ImmunoGeNN
unzip data_record.zip

pip install -r requirements.txt
```

Predicting protein immunogenicity risk scores:
```
python run.py --fasta_file \
    data/input.fasta
```

Deimmunizing first protein sequence:
```
python run.py \
    --fasta_file data/input.fasta \
    --deimmunize_first_sequence true
```

### Docker setup
Build Docker image:
```
docker build -t app-immunogenn .
```

Run Docker container:
```
docker run -v $(pwd)/data:/app/data -it app-immunogenn \
    python run.py --fasta_file data/input.fasta
```

### Speed benchmark

![img/runtime.png](img/runtime.png)

----

### Citation
```.bib
@inproceedings{
    hoie2025_immunogenn,
    title={ImmunoGe{NN}: Accelerating Early Immunogenicity Assessment for Generative Design of Biologics},
    author={Magnus Haraldson H{\o}ie and Birkir Reynisson and Paolo Marcatili and Jesper Ferkinghoff-Borg and Kasper Lamberth and Katharina L. Kopp and Morten Nielsen and Vanessa Isabell Jurtz},
    booktitle={EurIPS 2025 Workshop on SIMBIOCHEM},
    year={2025},
    url={https://openreview.net/forum?id=kOJQm9YXnB}
}
```

### License
This project is licensed under the MIT License - see the LICENSE file for details.

