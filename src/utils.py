import glob
import os
import pickle
import time

import biolib
import lz4.frame
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import src.processing
from bs4 import BeautifulSoup

def merge_html_files(merge_files, output_file, remove=True, verbose=False):

    # Check if there are files to merge
    if not merge_files:
        print("No files to merge.")
        return

    # Check present files. Use slice to copy list
    for file in merge_files[:]:
        if not os.path.exists(file):
            merge_files.remove(file)

            if verbose:
                print(f"Unable to find file: {file}")

    assert (
        len(merge_files) >= 1
    ), "No output plots successfully produced. This is likely a bug"

    # Read the content of the first file
    with open(merge_files[0], "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Iterate over the rest of the files and merge their content
    for file_path in merge_files[1:]:
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            soup_to_merge = BeautifulSoup(f, "lxml")

            # Merge the <head> sections
            head = soup.head
            head_to_merge = soup_to_merge.head
            for element in head_to_merge.find_all():
                if not head.find(
                    type=element.get("type", ""), src=element.get("src", "")
                ):
                    head.append(element)

            # Add linebreaks between merged files
            soup.body.append(soup.new_tag("br"))
            soup.body.append(soup.new_tag("br"))

            # Merge the <body> sections
            body = soup.body
            body_to_merge = soup_to_merge.body
            for element in body_to_merge.find_all(recursive=False):
                body.append(element)

    # Write the merged content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(soup))
    # Remove pre-merge files
    if remove:
        for file in merge_files:
            if os.path.exists(file):
                os.remove(file)

def merge_sav_files(sav_files, verbose=True):

    L = []
    for sav_file in sav_files:
        assert os.path.exists(sav_file), f"SAV file not found: {sav_file}"
        df = pd.read_csv(sav_file)
        L.append(df)

    # merge
    df_merged = L[0].copy()
    df_merged["in_reference"] = ""
    for df in L:
        # Update with max
        not_in_ref_m = df["in_reference"] != True
        m = (df["pIRS_rank"] >= df_merged["pIRS_rank"]) & not_in_ref_m
        df_merged.loc[m, "pIRS_rank"] = df.loc[m, "pIRS_rank"]
        df_merged.loc[m, "pIRS"] = df.loc[m, "pIRS"]
        df_merged.loc[m, "core_seq"] = df.loc[m, "core_seq"]


    # Normalize to DRB1 range for visualization
    irs = src.processing.normalize_y_hat_0_100_to_irs(
        df_merged["pIRS_rank"].values,
        model_dir="model/DRB1",
        verbose=verbose,
    )

    # Exclude in ref (pIRS 0)
    m = ~(df_merged["pIRS"] > 0)
    irs[m] = 0.0
    df_merged["pIRS"] = irs

    # 1000 separator
    N = len(df_merged) * len(sav_files)
    print(f"\nMerging {N:,} SAVs predicted across {len(sav_files)} gene class models")

    return df_merged


def merge_all_gene_pIRS_files(pIRS_files):

    df_merged = pd.DataFrame(
        columns=[
            "id",
            "peptide_pos",
            "peptide_seq",
            "pIRS",
            "pIRS_rank",
            "core_seq",
            "core_pos",
            "in_reference",
        ]
    )

    # Placeholder
    df_merged["pIRS_rank"] = 0.0
    df_merged["core_seq"] = ""

    for i, pIRS_file in enumerate(pIRS_files):

        df_gene = pd.read_csv(pIRS_file)
        missing_cols = [
            col
            for col in ["id", "peptide_pos", "peptide_seq", "pIRS", "pIRS_rank"]
            if col not in df_gene.columns
        ]
        assert not missing_cols, f"Missing columns in {pIRS_file}: {missing_cols}"

        if i == 0:
            update_cols = [
                "id", "peptide_pos", "peptide_seq",
                "pIRS", "pIRS_rank",
                "core_seq", 
                ]
            df_merged[["id", "peptide_pos", "peptide_seq"]] = df_gene[
                ["id", "peptide_pos", "peptide_seq"]
            ]
            df_merged["pIRS_rank"] = 0.0
            df_merged["pIRS"] = 0.0
        else:
            assert df_merged["id"].equals(df_merged["id"])
            assert df_merged["peptide_pos"].equals(df_merged["peptide_pos"])
            assert df_merged["peptide_seq"].equals(df_merged["peptide_seq"])

        # Get per-gene pIRS rank
        df_merged[f"pIRS_rank"] = df_gene["pIRS_rank"].values
        df_merged[f"in_reference"] = df_gene["in_reference"].values
        df_merged[f"core_seq"] = df_gene["core_seq"].values

        # Set rank, core to max scoring one
        m = (df_gene["pIRS_rank"] >= df_merged["pIRS_rank"].values).values
        
        # Present
        df_merged.loc[m, "core_seq"] = df_gene.loc[m, "core_seq"].values

        # Only update pIRS rank if not in reference
        m2 = m & (df_gene["in_reference"] != True).values
        df_merged.loc[m2, "pIRS_rank"] = df_gene.loc[m2, "pIRS_rank"].values
        
    # Get core pos
    core_pos = [
        pep.find(core)
        for pep, core in df_merged[["peptide_seq", "core_seq"]].values
    ]
    df_merged["core_pos"] = core_pos

    # Set in_reference to True if ANY gene classes are in_reference
    V = (
        (df_merged["in_reference"] == True)
    )
    df_merged["in_reference"] = V

    # Merged rank -> simulated DRB1 rank percentiles for interpretability
    pIRS = src.processing.normalize_y_hat_0_100_to_irs(
        df_merged["pIRS_rank"].values, model_dir="model/DRB1", verbose=False
    )

    # Exclude in ref
    m = df_merged["in_reference"] == True
    pIRS[m] = 0.0

    df_merged["pIRS"] = pIRS

    return df_merged

def filter_df_irs(
    df_irs,
    gene_class_str="DRB1",
    population_str="North America",
    peptide_class_str="1,2,3",
    filter_reference=False,
    verbose=False,
):

    gene_class_list = gene_class_str.split(",")
    population_list = population_str.split(",")
    peptide_class_list = [int(x) for x in peptide_class_str.split(",")]

    if verbose:
        print(f"gene_class_list: {gene_class_list}")
        print(f"population_list: {population_list}")
        print(f"peptide_class_list: {peptide_class_list}")
        print(f"filter_reference: {filter_reference}")

    # Read in
    # df_irs = pd.read_csv(irs_file, sep="\t")

    # Filter peptide_class
    df_irs = df_irs[df_irs["peptide_class"].isin(peptide_class_list)]

    # Remove in_reference
    if filter_reference:
        mask = df_irs["in_reference"] == True
        df_irs.loc[mask, "irs"] = 0

    # Filter gene_class (DRB1), population (Europe), peptide_class (1-3)
    df_irs = df_irs[df_irs["gene_class"].isin(gene_class_list)]

    # Filter population
    df_irs = df_irs[df_irs["population"].isin(population_list)]

    return df_irs

def prepare_output_pirs_file(pIRS_file):

    df_pirs = pd.read_csv(pIRS_file)

    # Rename 0, 1, 2, etc. to core_0, core_1, core_2, etc.
    core_cols = ["0", "1", "2", "3", "4", "5", "6"]
    new_core_cols = [f"core_{col}" for col in core_cols]
    df_pirs = df_pirs.rename(columns=dict(zip(core_cols, new_core_cols)))

    df_pirs["gene_class"] = "DRB1"

    order = ['id', 'peptide_pos',
            "gene_class",
            'peptide_seq',
            'core_pos',
            'core_seq', 
            'pIRS', 'pIRS_rank',
            'in_reference',
            'core_0', 'core_1', 'core_2', 'core_3', 'core_4', 'core_5', 'core_6',
        ]
    df_pirs = df_pirs[order]

    return df_pirs

def save_pirs_file_rounded(df_pirs, outfile):
    df_out = df_pirs.copy()

    core_cols = ["core_0", "core_1", "core_2", "core_3", "core_4", "core_5", "core_6"]
    df_out[core_cols] = df_out[core_cols].map(lambda x: f"{x:.2f}")
    df_out["pIRS"] = df_out["pIRS"].apply(lambda x: f"{x:.5f}")
    df_out["pIRS_rank"] = df_out["pIRS_rank"].apply(lambda x: f"{x:.3f}")

    df_out.to_csv(outfile, index=False)

def irs_file_to_df_seq(
    irs_file,
    peptide_class_str="1,2,3,4",
    population_str="North America",
    gene_class_str="DRB1",
    filter_reference=False,
    verbose=False,
):
    """
    Convert IRS file to one score per peptide Immunogenn format (similar to scores.tsv)
    Considers only peptide_class 1,2,3, gene_class DRB1, population North America
    """

    df_irs = pd.read_csv(irs_file, sep="\t")

    required_cols = [
        "peptide_pos",
        "peptide_seq",
        "population",
        "irs",
        "peptide_class",
        "in_reference",
    ]
    assert all(
        pd.Series(required_cols).isin(df_irs.columns)
    ), f"Missing columns in {irs_file}"

    # Prepare df with one score per peptide
    df_irs_filtered = filter_df_irs(
        df_irs,
        peptide_class_str=peptide_class_str,
        population_str=population_str,
        gene_class_str=gene_class_str,
        filter_reference=filter_reference,
        verbose=verbose,
    )

    Ls = []
    peptide_pos = sorted(df_irs_filtered["peptide_pos"].unique())
    for pos in peptide_pos:
        df = df_irs_filtered[df_irs_filtered["peptide_pos"] == pos]
        pep_sum_irs = df["irs"].sum()
        Ls.append([pos, pep_sum_irs])

    # One score per peptide
    df_pep_sum_irs = pd.DataFrame(Ls, columns=["peptide_pos", "irs"])

    # Prepare output df
    df_out = df_irs.drop_duplicates(subset=["peptide_pos", "peptide_seq"])
    df_out = df_out[["id", "peptide_pos", "peptide_seq", "in_reference"]].reset_index(
        drop=True
    )
    df_out["irs"] = 0.0

    # Assign values from one score per peptide
    df_out.loc[df_pep_sum_irs["peptide_pos"], "irs"] = df_pep_sum_irs["irs"].values
    df_out.index = df_out["id"].values
    df_out["peptide_pos"] += 1

    return df_out

def save_df_fasta_to_pIRS_scores_csv(df_fasta, outdir, population="Global", save=True):

    os.makedirs(outdir, exist_ok=True)

    # Scores 6 point precision
    S_scores = df_fasta.groupby("id", sort=False)["pIRS"].sum()
    df_scores = pd.DataFrame(
        {
            "id": S_scores.index,
            "population": f"{population}",
        }
    )

    df_scores.index = df_scores["id"]

    gene_classes = ["DRB1"]
    for gene_class in gene_classes:

        col = f"pIRS_rank"
        if not col in df_fasta.columns:
            continue

        # Already weighted
        scores = src.processing.normalize_y_hat_0_100_to_irs(df_fasta[col].values, model_dir=f"model/{gene_class}/peptide", verbose=False)

        for id in df_fasta["id"].unique():
            m = (df_fasta["id"] == id).values
            pIRS_weighted_sum = scores[m].sum()
            df_scores.loc[id, f"{gene_class}_pIRS_sum"] = pIRS_weighted_sum

    # Save
    if save:
        # Round to 6 point precision
        numerical_cols = df_scores.select_dtypes(include=[np.number]).columns
        df_scores[numerical_cols] = df_scores[numerical_cols].map(
            lambda x: f"{x:.5f}"
        )
        df_scores.to_csv(f"{outdir}/scores.csv", index=False)
        print(f"Writing summary pIRS file to {outdir}/scores.csv\n")

    return df_scores


def df_fasta_to_df_seq(df_fasta, record):
    df_seq = df_fasta[df_fasta["id"] == record.id]
    return df_seq


def get_df_seq_top_peptide_positions(df_seq, threshold_irs=0.097):

    # Get top peptides
    mask = df_seq["pIRS"] >= threshold_irs
    positions = df_seq.loc[mask, "peptide_pos"].values

    # Construct list of 15mer positions, non-overlapping
    pos_list = []
    seen_set = set()
    for pos in positions:

        if pos not in seen_set:
            start = pos
            end = pos + 15
            seen_set.update(range(start, end))
            pos_list.append(list(range(start, end)))

    return pos_list


def extract_df_seq_sequence_id(df_fasta):
    """
    Extract the sequence from a df_fasta
    """
    seq1 = "".join(df_fasta.loc[:, "peptide_seq"].apply(lambda s: s[0]))
    seq2 = df_fasta.iloc[-1]["peptide_seq"][1:]
    seq = seq1 + seq2
    id = df_fasta.iloc[0]["id"]
    return seq, id


def get_df_heatmap_wt_scores(df_heatmap):
    sequence = df_heatmap.index
    # aa_columns = df_heatmap.columns
    scores = []
    for i, wt_aa in enumerate(sequence):
        scores.append(df_heatmap.iloc[i, df_heatmap.columns.get_loc(wt_aa)])

    return pd.Series(scores, index=sequence)


def get_seq_esm_LLR_dataframe(sequence, esm_model="esm2_t6_8M_UR50D", verbose=True):
    """
    Get the ESM-2 per-position log-probabilities as a DataFrame for the provided sequence using the esm2_t6 model.
    """

    if verbose:
        print(f"Loading ESM-2 model {esm_model} ...")

    # Load the pretrained model and its alphabet for the specified model version
    import esm.pretrained
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model)
    batch_converter = alphabet.get_batch_converter()

    # Prepare the data (ESM expects the input as a batch)
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass to get the logits
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])

    # Logits for the tokens (ignore the [CLS] prepended token and exclude [EOS])
    logits = results["logits"].squeeze(0)[1:-1]

    # Calculate log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Convert to DataFrame
    amino_acids = (
        alphabet.all_toks
    )  # Remove special starting ([CLS], PAD) and end ([EOS]) tokens
    df = pd.DataFrame(log_probs.numpy(), columns=amino_acids, index=list(sequence))

    amino_acids = list("CWGMPSTDLEFQNAHVYRIKX")
    df = df[amino_acids]

    # Show as LLR versus WT
    # for i in range(len(df)):
    #     row = df.iloc[i]
    #     wt_aa = row.name
    #     LLR = (row - row[wt_aa]).values
    #     df.iloc[i] = LLR

    return df


def get_logprobs_top_residues(df_log_probs, n=4):
    """
    Get the top N residues for each position in the provided DataFrame of ESM-2 log-probabilities.
    """
    top_residues = df_log_probs.apply(lambda x: x.nlargest(n).index.tolist(), axis=1)
    df_top_residues = pd.DataFrame(top_residues.tolist(), index=top_residues.index)
    return df_top_residues


def get_bottom_residues(df_log_probs, n=4):
    """
    Get the bottom N residues for each position in the provided DataFrame of ESM-2 log-probabilities.
    """
    bottom_residues = df_log_probs.apply(
        lambda x: x.nsmallest(n).index.tolist(), axis=1
    )
    df_bottom_residues = pd.DataFrame(
        bottom_residues.tolist(), index=bottom_residues.index
    )
    return df_bottom_residues


def seq_to_df_fasta_savs(seq, _id):

    all_15mers_list = []
    all_15mers_ids_list = []
    start_list = []

    for pos in range(len(seq) - 14):

        # Extract the 15mer
        original_15mer = seq[pos : pos + 15]

        # Generate all SAVs of the 15mer
        for rel_pos in range(15):
            aa_orig = original_15mer[rel_pos]

            for aa in "ACDEFGHIKLMNPQRSTVWY":
                sav_15mer = (
                    original_15mer[:rel_pos] + aa + original_15mer[rel_pos + 1 :]
                )
                all_15mers_list.append(sav_15mer)
                all_15mers_ids_list.append(
                    f"{_id}___{aa_orig}{pos+rel_pos+1}{aa}___{rel_pos}"
                )
                start_list.append(pos + 1)

    # DataFrame, dropping duplicates
    df = pd.DataFrame(data={"id": all_15mers_ids_list, "peptide_seq": all_15mers_list, "start": start_list})
    df["end"] = df["start"] + 14

    return df


def get_gene_class_from_model_dir(model_dir):
    gene_class = ""
    if "DRB1" in model_dir:
        gene_class = "DRB1"
    else:
        raise Exception(f"Warning: Unable to determine gene class from model path: {model_dir}")

    return gene_class


def load_references(human_references_pkl="", extra_references="", verbose=True):

    def load(infile):
        with lz4.frame.open(infile, "rb") as f:
            return pickle.load(f)

    def dump(outfile, d):
        with lz4.frame.open(outfile, "wb", compression_level=0) as f:
            pickle.dump(d, f, protocol=5)

    def parse_fasta_to_9mer_set(fasta_file):
        records_gen = biolib.utils.SeqUtil.parse_fasta(fasta_file)

        # parse into 9mers into dict
        input_peptides_set = set()
        for record in records_gen:
            seq = record.sequence
            if len(seq) >= 9:
                for i in range(len(seq) - 9 + 1):
                    input_peptides_set.add(seq[i : i + 9])

        return input_peptides_set

    if verbose:
        print(
            f"\nLoading human references file (131 MB) from {human_references_pkl} ..."
        )

    assert os.path.exists(human_references_pkl), f"Unable to locate human reference file {human_references_pkl}. Please run 'unzip data_record.zip'"

    # Human refs pre-computed in csv
    references_set = set()
    if human_references_pkl:
        assert os.path.exists(
            human_references_pkl
        ), f"Unable to locate human reference file {human_references_pkl}. Please run 'unzip data_record.zipf'"
        # references_set = pickle.load(open(human_references_pkl, "rb"))
        references_set = load(human_references_pkl)

    # Extra references compute to CSV later
    if extra_references:
        assert os.path.exists(
            extra_references
        ), f"Unable to locate reference file: {extra_references}"
        print(f"\nLoading uploaded reference file {extra_references}")
        input_peptides_set = parse_fasta_to_9mer_set(extra_references)
        references_set.update(input_peptides_set)

    return references_set


def predict_df_fasta(
    model_list,
    df_fasta,
    human_refs_set=False,
    normalize=True,
    outname="",
    outdir="output/",
    save=False,
    verbose=False,
    invert_peptides=False,
):

    # need outname if save true
    if save:
        assert outname, "Need output filename if save is True"

    start_time = time.time()

    df_fasta0 = df_fasta.copy()

    if invert_peptides:
        print(f"Inverting {len(df_fasta0)} peptides for prediction ...")
        print(f"Before: {df_fasta0['peptide_seq'].tolist()[:5]}")
        df_fasta0["peptide_seq"] = df_fasta0["peptide_seq"].apply(lambda s: s[::-1])
        print(f"After: {df_fasta0['peptide_seq'].tolist()[:5]}")

    X_input, df_try = (
        src.processing.preprocess_df_fasta_binding_cores(df_fasta0)
    )

    df_fasta_list = []
    df_outfile_list = []
    N_variants = len(df_fasta0)
    for i, model_dir in enumerate(model_list):

        gene_class = get_gene_class_from_model_dir(model_dir)
        print()
        print(
            f"{gene_class}: Predicting {N_variants:,} peptides with model {i+1}/{len(model_list)}: {model_dir} ..."
        )

        df_fasta = src.processing.predict_df_fasta_cores_and_pep(
            model_dir=model_dir,
            df_fasta=df_fasta0,
            X_input=X_input,
        )

        # Filter references (set score to 0.00)
        df_fasta["in_reference"] = ""
        if human_refs_set:
            start = time.time()

            if "core_seq" in df_fasta.columns:
                unique_cores = set(df_fasta["core_seq"])
                overlapping_cores = unique_cores.intersection(human_refs_set)
                m = df_fasta["core_seq"].isin(overlapping_cores)

            df_fasta.loc[m, ["in_reference", "pIRS", "pIRS_rank"]] = [True, 0.0, 0.0]

        # Prepare outname
        outfile = f"{outdir}/{gene_class}/{outname}"
        df_outfile_list.append(outfile)

        if save:
            print(f"\tSaving to {outfile} ...")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            df_out = df_fasta.copy()
            df_out["pIRS"] = df_out["pIRS"].apply(lambda x: f"{x:.5f}")
            df_out["pIRS_rank"] = df_out["pIRS_rank"].apply(lambda x: f"{x:.3f}")
            df_out.to_csv(f"{outfile}", index=False)

    if verbose:
        print(
            f"\nPredicted {len(df_fasta['ID'].unique())} peptides in {time.time() - start_time:.2f} seconds\n"
        )

    return df_fasta_list, df_outfile_list


def create_single_gene_pIRS_files(pIRS_list, outdir, remove=False):
    """Writes merged pIRS file per gene folder"""
    pIRS_S = pd.Series(pIRS_list)
    assert pIRS_S.str.contains(
        f"{outdir}"
    ).all(), f"pIRS files not in {outdir}: {pIRS_S}"
    assert pIRS_S.str.contains(
        f"__"
    ).all(), f"pIRS files not following __ format: {pIRS_S}"

    merged_files = []
    gene_folders = set([os.path.dirname(f) for f in pIRS_S])
    for gene_folder in gene_folders:
        gene_files = pIRS_S[pIRS_S.str.contains(gene_folder)].to_list()
        df_list = [pd.read_csv(f) for f in gene_files]
        df_merged = pd.concat(df_list, axis=0)

        outname = f"{gene_folder}/pIRS.csv"
        df_merged.to_csv(outname, index=False)
        print(f"Writing merged pIRS file to {outname}")

        merged_files.append(outname)

        # Remove gene_files
        if remove:
            for f in gene_files:
                os.remove(f) if os.path.exists(f) else None

    return merged_files

def model_predict_pirs_and_rank(
    model_path, X_input, model_dir="model/", normalize=True, cap=False, verbose=True
):
    """ """

    if model_path.endswith(".onnx"):
        import onnxruntime as rt

        # Load ONNX model (do once at startup)
        onnx_session = rt.InferenceSession(model_path)
        input_name = onnx_session.get_inputs()[0].name

        # Fast predict
        X_input_float32 = X_input.astype(np.float32)
        y_pred = onnx_session.run(None, {input_name: X_input_float32})[0]

    else:
        assert os.path.exists(model_path), f"Unable to locate model file: {model_path}"
        model = pickle.load(open(model_path, "rb"))
        y_pred = model.predict(X_input)

    # First rank, then IRS scale
    pirs_rank = src.processing.normalize_y_hat_to_0_100(
        y_pred, model_dir=model_dir, verbose=False
    )
    pirs = src.processing.normalize_y_hat_0_100_to_irs(
        pirs_rank, model_dir=model_dir, verbose=False
    )

    if cap:
        pirs = np.clip(pirs, 0.0, 1.300959)  # 99.5% percentile of pIRS (y_hat -> )
        print(
            f"Normalizing y_hat to pIRS and pIRS%, capping to 99.5 percentile: {1.300959}"
        )

    return pirs, pirs_rank
