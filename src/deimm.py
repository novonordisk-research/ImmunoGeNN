import pandas as pd
import numpy as np

import src
import src.utils
import biolib

def df_fasta_savs_to_df_mut(df_fasta_savs, record, esm_model="esm2_t6_8M_UR50D"):
    # Plot SAVs all
    df_heatmap = src.plots_pred.df_fasta_savs_to_2d_heatmap(
        df_fasta_savs, record.sequence, pIRS_col="pIRS", norm=False
    )

    df_heatmap_rank = src.plots_pred.df_fasta_savs_to_2d_heatmap(
        df_fasta_savs, record.sequence, pIRS_col="pIRS_rank", norm=False
    )

    # Save heatmap
    df_out = df_heatmap_rank.copy()
    # Add aa indices
    indices = [
        f"{i}_{aa}" for aa, i in zip(df_out.index, range(1, len(df_out.index) + 1))
    ]
    df_out.index = indices

    # Drop cols
    df_out = df_out.drop(columns=["X"])

    # Plot SAVs + ESM plot
    # df_heatmap_filtered = df_heatmap.copy()
    df_heatmap_esm = src.utils.get_seq_esm_LLR_dataframe(record.sequence, esm_model=esm_model)
    df_heatmap_esm_norm = pd.Series(df_heatmap_esm.values.flatten()).rank(pct=True).values.reshape(df_heatmap_esm.shape)

    mut_dict = {}
    for i in range(len(df_heatmap)):
        row = df_heatmap.iloc[i]
        pos = i +1
        wt = row.name

        for j in range(len(row)):

            mut = row.index[j]
            mut_str = f"{wt}{pos}{mut}"

            if mut == "X":
                continue

            pIRS = df_heatmap.iloc[i, j]
            wt_pIRS = df_heatmap.iloc[i][df_heatmap.iloc[i].name]

            pIRS_rank = df_heatmap_rank.iloc[i, j]
            wt_pIRS_rank = df_heatmap_rank.iloc[i][df_heatmap_rank.iloc[i].name]

            delta = pIRS - wt_pIRS
            esm = df_heatmap_esm.iloc[i, j]

            d = {f"{mut_str}": {
                "pIRS": pIRS,
                "wt_pIRS": wt_pIRS,
                "pIRS_rank": pIRS_rank,
                "wt_pIRS_rank": wt_pIRS_rank,
                "delta": delta,
                "esm": esm,
                "pos": pos,
                "mut_str": mut_str,
            }}

            mut_dict.update(d)

    df_mut = pd.DataFrame.from_dict(mut_dict, orient="index")
    return df_mut

def weight_df_mut(df_mut, only_deimmunizing=True):
    df_out_rank = df_mut.copy()
    df_out_rank["esm_rank"] = df_out_rank["esm"].rank(pct=True)
    df_out_rank["weight"] = df_out_rank["delta"] * (df_out_rank["esm_rank"]**2)

    # Sort
    df_out_rank = df_out_rank.sort_values(by="weight", ascending=True)

    # Only deimmunizing
    if only_deimmunizing:
        df_out_rank = df_out_rank[df_out_rank["delta"] < 0]

    return df_out_rank

def mut_str_to_seq(mut_str, seq):
    pos = int(mut_str[1:-1]) - 1
    wt = mut_str[0]
    mut = mut_str[-1]
    assert seq[pos] == wt, f"mut_str {mut}{pos} != input sequence {seq[pos]}{pos}"

    new_seq = seq[:pos] + mut + seq[pos+1:]
    return new_seq

def df_mut_weighted_to_fasta_records(df_mut_weighted, orig_record, top_n=20, verbose=True):

    records = [orig_record]
    for i in range(len(df_mut_weighted.iloc[0:top_n])):

        row = df_mut_weighted.iloc[i]
        mut_str = row['mut_str']
        seq = mut_str_to_seq(mut_str, orig_record.sequence)

        # floor
        wt_pirs = np.floor(row['wt_pIRS_rank'])
        pirs = np.floor(row['pIRS_rank'])

        id = f"{orig_record.id}__{i+1}_{mut_str}_{wt_pirs:.0f}_to_{pirs:.0f}"
        desc = f"Wild-type pIRS% {wt_pirs:.2f} -> {pirs:.2f}, ESM2 {row['esm']:.2f} (logp)"
        R = biolib.utils.seq_util.SeqUtilRecord(sequence_id=id, sequence=seq, description=desc)
        records.append(R)
    
        if verbose:
            print(f"{id}: {desc}")

    return records
