import colorsys
import pickle

# import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Plotly theme simple_white
pio.templates.default = "simple_white"
import joblib
from scipy.spatial.distance import pdist, squareform

import src.utils


def plot_deimmunization_plot(
    df_fasta_savs,
    record,
    gene_class="",
    top_n=10,
    top_n_esm=-1,
    esm_model="esm2_t6_8M_UR50D",
    remove_cysteines=False,
    only_deimmunizing=True,
    width=1000,
    height=475,
    save=True,
):

    if top_n_esm == -1:
        top_n_esm = top_n

    print(top_n)

    df_fasta_savs = df_fasta_savs.copy()

    # Plot SAVs all
    df_heatmap = df_fasta_savs_to_2d_heatmap(
        df_fasta_savs, record.sequence, pIRS_col="pIRS", norm=False
    )

    df_heatmap_rank = df_fasta_savs_to_2d_heatmap(
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

    # for i in range(len(df_heatmap)):
    #     row = df_heatmap.iloc[i]
    #     wt_score = row.loc[row.name]

    #     order = np.argsort(row.values)
    #     max_v = min(row[order[top_n]], wt_score)
    #     m1 = row >= max_v
    #     m2 = row.index != row.name
    #     mc = (m1 & m2)
    #     df_heatmap.iloc[i, mc] = -1

    if only_deimmunizing and top_n < 20:
        print(f"Only showing top {top_n} deimmunizing mutations")
        for i in range(len(df_heatmap)):
            row = df_heatmap.iloc[i]
            wt_score = row.loc[row.name]

            order = np.argsort(row.values)
            # Use .iloc for positional access to avoid FutureWarning
            threshold_pos = order[top_n]
            threshold_value = row.iloc[threshold_pos]
            max_v = min(threshold_value, wt_score)

            m1 = row >= max_v
            m2 = row.index != row.name
            mc = m1 & m2
            df_heatmap.iloc[i, mc] = -1

    # Plot SAVs + ESM plot
    # df_heatmap_filtered = df_heatmap.copy()
    seq_df_probs = src.utils.get_seq_esm_LLR_dataframe(record.sequence, esm_model)
    seq_df_probs_top_n = src.utils.get_logprobs_top_residues(seq_df_probs, n=top_n_esm)

    # Mask below top n
    for i2 in range(len(seq_df_probs_top_n)):
        top_n_aas = seq_df_probs_top_n.iloc[i2].values
        mask = ~df_heatmap.columns.isin(top_n_aas)
        df_heatmap.iloc[i2, mask] = -1

    # Only show top 10 deimmunizing mutations
    # m = np.argsort(-df_heatmap, axis=1) > top_n
    # df_heatmap[m] = -1

    # Plot only ESM suggested mutations
    fig = plot_df_heatmap_deimmunizing_mutations(
        df_heatmap,
        df_heatmap_rank,
        record.sequence,
        seq_df_probs,
        name=record.id,
        remove_cysteines=remove_cysteines,
        cmap_name="viridis_r",
        width=width,
        height=height,
    )

    N_variants = len(df_fasta_savs)
    title = f"Top {top_n} deimmunizing variants across {record.id} (of {N_variants:,} screened)<br><sup>(Maximum peptide score for given residue mutation)</sup>"
    # title2 = f"Maximum residue score after screening {N_mutations} possible deimmunizing mutations in {record.id}<br><sup>(Maximum peptide score for given residue mutation)</sup>""
    if gene_class:
        title = f"{gene_class} " + title

    fig.update_layout(title_text=title)

    return fig


def plot_df_heatmap_deimmunizing_mutations(
    df_heatmap,
    df_heatmap_rank,
    seq,
    df_heatmap_esm=False,
    name="the sequence",
    remove_cysteines=False,
    cmap_name="tab20_r",
    y_min=-0.1,
    y_max=1.1,
    width=1250,
    height=500,
    percentiles_85_95_99=(),
):
    """ """

    if len(percentiles_85_95_99) != 3:

        try:
            # print(f"Trying to load percentiles from {model_dir}/quantilenormalizer.pkl")
            qt = joblib.load(f"model/DRB1/quantilenormalizer.pkl")
            values = [[0.85], [0.95], [0.99]]
            q85, q95, q99 = qt.inverse_transform(values).reshape(-1)
            percentiles_85_95_99 = np.round([q85, q95, q99], 3)
            print(f"pIRS percentiles 85%, 95%, 99% are {percentiles_85_95_99}")

        except Exception as e:
            percentiles_85_95_99 = [0.088903, 0.285847, 1.017301]
            print(
                f"pIRS percentiles not provided. Assigning 85%, 95%, 99% as {percentiles_85_95_99}"
            )

    # QGHD: Exclude cysteines - liability never want to mutate to
    if remove_cysteines:
        mask = df_heatmap["C"].index != "C"
        df_heatmap.loc[mask, "C"] = -1.00

    # Process
    df_melted = process_heatmap(df_heatmap, seq)

    # Hack: Always negative values
    df_melted["Delta"] = df_melted["Delta"].abs()
    df_melted["Delta"] = df_melted["Delta"] * 100

    # Add rank percentiles
    df_melted_rank = process_heatmap(df_heatmap_rank, seq)
    # df_melted_rank = df_melted_rank * 100

    # Hack: Removing missing values
    m1 = df_melted.isna().sum(axis=1) > 1
    m2 = df_melted_rank.isna().sum(axis=1) > 1
    mc = (m2).values
    df_melted = df_melted.loc[~mc]
    df_melted_rank = df_melted_rank.loc[~mc]

    df_melted["pIRS_rank"] = df_melted_rank["Score"].values
    df_melted["pIRS"] = df_melted["Score"].values
    df_melted["mut_str"] = df_melted["index"] + df_melted["Amino Acid"]

    m_wt = ~(df_melted["not_wt_mask"].values)
    df_melted.loc[m_wt, "mut_str"] = df_melted.loc[m_wt, "mut_str"].apply(
        lambda s: f"Wiltype ({s})"
    )

    # Custom data to plot
    custom_data_list = [
        "Amino Acid",
        "peptide_pos",
        "mut_str",
        "Delta",
        "wt_pos_mut",
        "not_wt_mask",
        "pIRS_rank",
        "pIRS",
    ]

    # Add ESM values if present
    if isinstance(df_heatmap_esm, pd.DataFrame):
        df_melted_esm = process_heatmap(
            df_heatmap_esm,
            seq,
            score_below_n_to_nan=False,
            scores_below_wt_to_nan=False,
        )
        df_melted.loc[df_melted.index, "ESM_LLR"] = df_melted_esm.loc[
            df_melted.index, "Score"
        ].round(4)
        custom_data_list.append("ESM_LLR")

    # Custom colors
    amino_acids = list("CWGMPSTDLEFQNAHVYRIKX")
    color_map = create_custom_colormap(cmap_name, amino_acids)

    # Everything above threshold red
    percentile_95 = percentiles_85_95_99[-2]
    red_mask = df_melted["Score"] >= percentile_95
    df_melted["display_color"] = df_melted["Amino Acid"].values

    # Light green for > 85
    for i in range(0, 20):
        color_map[f"Green{i}"] = "#90EE90"
    for pos in df_melted["peptide_pos"].unique():
        df = df_melted[df_melted["peptide_pos"] == pos]
        mask = (df["Score"] < percentiles_85_95_99[0]).values
        labels = [f"Green{i}" for i in range(mask.sum())]
        df_melted.loc[df.index[mask], "display_color"] = labels

    # Orange for >= 85
    for i in range(0, 20):
        color_map[f"Orange{i}"] = "#FFA500"
    for pos in df_melted["peptide_pos"].unique():
        df = df_melted[df_melted["peptide_pos"] == pos]
        mask = (df["Score"] >= percentiles_85_95_99[0]).values
        labels = [f"Orange{i}" for i in range(mask.sum())]
        df_melted.loc[df.index[mask], "display_color"] = labels

    # Red 95
    for i in range(0, 20):
        color_map[f"Red{i}"] = "#FA8072"
    for pos in df_melted["peptide_pos"].unique():
        df = df_melted[df_melted["peptide_pos"] == pos]
        mask = (df["Score"] >= percentiles_85_95_99[1]).values
        labels = [f"Red{i}" for i in range(mask.sum())]
        df_melted.loc[df.index[mask], "display_color"] = labels

    # First plot red only
    df_melted["pos"] = df_melted["pos"].astype(str)
    original_order = df_melted["pos"].unique()

    fig_scatter = px.scatter(
        df_melted,
        x="pos",
        y="Score",
        color="display_color",
        custom_data=custom_data_list,
        color_discrete_map=color_map,
        category_orders={"pos": original_order},
    )

    fig_scatter.update_traces(
        marker=dict(size=8, line=dict(color="black", width=0.75)),
        # hovertemplate="%{customdata[2]} to <b>%{customdata[0]}</b>: %{y:.3f} (%{customdata[3]:.0f}%)  <extra></extra>",
        # hovertemplate=" %{customdata[2]}<b>%{customdata[0]}</b>: pIRS %{customdata[6]:.2f}% <extra></extra>",
        hoverlabel=dict(align="auto"),
    )

    # Remove legend for Red
    fig_scatter.for_each_trace(
        lambda trace: trace.update(showlegend=False) if "Red" in trace.name else ()
    )
    fig_scatter.for_each_trace(
        lambda trace: trace.update(showlegend=False) if "Orange" in trace.name else ()
    )
    fig_scatter.for_each_trace(
        lambda trace: trace.update(showlegend=False) if "Green" in trace.name else ()
    )

    # ESM labels if present
    if isinstance(df_heatmap_esm, pd.DataFrame):
        fig_scatter.update_traces(
            # hovertemplate=" %{customdata[2]}<b>%{customdata[0]}</b>: pIRS %{customdata[6]:.2f}%, ESM2 probability %{customdata[8]:.2f} (logP)  <extra></extra>",
            hovertemplate=" %{customdata[2]}: pIRS %{customdata[6]:.2f}%, ESM2 probability %{customdata[8]:.2f} (logP)  <extra></extra>",
        )

    # Reverse the order of the legend
    fig_scatter.update_layout(legend=dict(orientation="v", traceorder="reversed"))

    # Create subplot and add traces
    fig = make_subplots(rows=1, cols=1)
    for trace in fig_scatter.data:
        fig.add_trace(trace, row=1, col=1)

    # Preserve order
    original_order = fig_scatter.layout.xaxis.categoryarray
    fig.update_xaxes(categoryorder="array", categoryarray=original_order)

    # Update layout and hover mode
    n_kmers = len(seq) - 14
    fig.update_layout(
        title_text=f"Variants with lowered immunogenicity across {name}<br><sup>Average residue score across all overlapping 15mers. Considers {n_kmers}x15x20 = {15*20*n_kmers} peptides<br></sup>",
        width=width,
        height=height,
        xaxis_title="Residue position",
        yaxis_title="pIRS",
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="v",
            traceorder="reversed",
            title=dict(text="Mutation to", side="top"),
            # Shift to the right a little
            x=1.040,
        ),
        hovermode="x",
        hoverdistance=6,
        hoverlabel=dict(align="right"),
    )
    fig.update_xaxes(spikemode="across", showspikes=True)

    # Adjust x ticks
    fig = plotly_add_xtick_intervals_seq(fig, seq)

    # Add sequence annotations x-axis
    wt_scores = get_df_heatmap_wt_scores(df_heatmap)
    percentile_95 = percentiles_85_95_99[-2]
    red_mask = wt_scores >= percentile_95
    fig = plotly_add_seq_xaxis(fig, seq, red_mask)

    # Add percentage y axis
    y_max = df_melted["Score"].max() + 0.2
    fig = plotly_secondary_y_axis(
        fig,
        yticks=percentiles_85_95_99,
        ytick_labels=["85%", "95%", "99%"],
        y_min=y_min,
        y_max=y_max,
    )

    # For 15mers, adjust plot
    if len(seq) == 15:
        fig.update_traces(marker=dict(size=12, line=dict(color="black", width=1.25)))

        # Update title text
        fig.update_layout(
            title_text=f"Deimmunizing mutations across {name} ({seq})<br><sup>Considering 15 x 20 SAVs across {n_kmers} x 15mers = {15*20*n_kmers} peptides<br></sup>"
        )

    # Auto layout
    fig.update_layout(
        autosize=True,
        width=width,
        height=height,
    )

    return fig


def get_df_heatmap_wt_scores(df_heatmap):
    wt_scores = np.array(
        [
            df_heatmap.iloc[pos, df_heatmap.columns.get_loc(aa)]
            for pos, aa in zip(range(len(df_heatmap)), df_heatmap.index)
        ]
    )
    return wt_scores


def create_custom_colormap(cmap_name, amino_acids):
    # import matplotlib.cm as cm
    # cmap = cm.get_cmap(cmap_name, len(amino_acids))
    # # Dump to cmap.pkl
    # with open("data/cmap2.pkl", "wb") as f:
    #     pickle.dump(cmap, f)

    cmap = pickle.load(open("data/cmap2.pkl", "rb"))
    custom_colors = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    color_map = {aa: custom_colors[i] for i, aa in enumerate(amino_acids)}
    return color_map


def process_heatmap(
    df_heatmap, seq, score_below_n_to_nan=-0.99, scores_below_wt_to_nan=True
):
    """
    Process the heatmap DataFrame by clipping values based on sequence length
    and only showing values below the wildtype score.
    """

    def melt_and_clean_dataframe(df_plot, score_below_n_to_nan=False):
        df_melted = (
            df_plot.reset_index().melt(
                id_vars="index", var_name="Amino Acid", value_name="Score"
            )
            # .dropna()
        )

        if score_below_n_to_nan:
            mask = df_melted["Score"] < score_below_n_to_nan
            df_melted.loc[mask, "Score"] = np.nan

        # Custom label data
        df_melted["peptide_pos"] = df_melted["index"].str[1:]

        # Set index to wt pos mut format (M1V)
        df_melted.index = df_melted["index"] + df_melted["Amino Acid"]

        return df_melted

    def calculate_deltas(df_melted):
        for pos in df_melted["peptide_pos"].unique():
            df = df_melted[df_melted["peptide_pos"] == pos]
            # delta = ((df["Score"] - df["Score"].max()) / df["Score"].max()) * 100
            delta = (df["Score"] - df["Score"].max()) / df["Score"].max()
            df_melted.loc[df.index, "Delta"] = delta
        return df_melted

    amino_acids = list("CWGMPSTDLEFQNAHVYRIKX")
    indices = pd.Series(df_heatmap.index).apply(lambda s: f"{s}") + np.arange(
        1, len(df_heatmap) + 1
    ).astype(str)

    df_plot = pd.DataFrame(
        data=df_heatmap.values.copy(),
        columns=amino_acids,
        index=indices,
    )
    df_plot = df_plot.astype(float)

    # If 15mer cap values at 0.00-0.10, otherwise to 0.08
    # if len(seq) == 15:
    #     df_plot = df_plot.clip(upper=0.11)
    # else:
    #     df_plot = df_plot.clip(upper=0.075)

    # Only show values below the wildtype score
    if scores_below_wt_to_nan:
        for i in range(len(df_plot)):
            row = df_plot.iloc[i]
            wt_aa = seq[i]
            wt_score = row[wt_aa]
            mask = (row > wt_score).values
            row[mask] = np.nan

    # Melt DataFrame for plotting
    df_melted = melt_and_clean_dataframe(
        df_plot, score_below_n_to_nan=score_below_n_to_nan
    )

    # Add deltas
    df_melted = calculate_deltas(df_melted)

    # Add wt_pos_mut
    df_melted["wt_pos_mut"] = df_melted["index"] + df_melted["Amino Acid"]
    df_melted["not_wt_mask"] = (
        df_melted["index"].apply(lambda s: s[0]) != df_melted["Amino Acid"]
    )

    # Add and sort by pos
    df_melted["pos"] = df_melted["wt_pos_mut"].apply(lambda s: s[1:-1])
    df_melted["pos"] = df_melted["pos"].astype(int)
    df_melted = df_melted.sort_values("pos")

    return df_melted


def plotly_secondary_y_axis(
    fig,
    yticks,
    ytick_labels,
    y_min=-0.00425,
    y_max=0.010,
):
    """
    Configure a secondary y-axis with custom tick values and labels on a Plotly figure.
    """

    fig.update_layout(yaxis_range=[y_min, y_max])

    # Custom y-axis
    x = [0]
    y = [1]
    fig.add_trace(
        go.Scatter(
            x=[min(x), max(x)],
            y=[min(y), max(y)],
            name="",
            yaxis="y2",
            line=dict(color="white", width=0, dash="dash"),
        )
    )

    # Hide original y axis
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        ticks="",
        title_text="",
        # Disable double-click
        fixedrange=False,
    )

    fig.update_layout(
        yaxis2=dict(
            title="pIRS",  # Optional: add a title if needed
            overlaying="y",  # Overlaying on the main y-axis
            side="left",  # Position on the right side of the plot
            tickvals=yticks,  # Set custom y-tick values
            ticktext=ytick_labels,  # Set custom labels for the y-ticks
            range=[y_min, y_max],  # Set the range for the y-axis
            fixedrange=False,  # Disable zooming on the secondary y-axis
        ),
        # Disable double-click
        yaxis=dict(fixedrange=False),
    )

    for i in range(len(yticks)):
        color = "black" if i < len(yticks) - 1 else "black"
        fig.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=yticks[i],
            y1=yticks[i],
            line=dict(color=color, width=1.00, dash="dash"),
        )

    return fig


def plotly_add_xtick_intervals_seq(fig, seq):
    def get_tickvals(seq, max_steps=11):
        step_sizes = [1, 5, 10, 20, 50, 100, 200]

        for step in step_sizes:
            tickvals = np.arange(0, len(seq) + 1, step)
            if len(tickvals) <= max_steps:
                return tickvals

        return np.arange(0, len(seq) + 1, step_sizes[-1])

    tickvals = get_tickvals(seq)
    fig.update_xaxes(tickvals=tickvals)

    return fig


def plotly_add_seq_xaxis(fig, seq, red_mask=None, red_only=False):

    annotations = []

    # Add red above threshold
    if red_mask is not None:
        for i, aa in enumerate(seq):
            color = "red" if red_mask[i] else "black"
            y = 0.012 if red_mask[i] else 0.01
            aa = f"<b>{aa}</b>" if red_mask[i] else aa

            if color == "black" and red_only:
                continue

            annotations.append(
                dict(
                    x=i,
                    y=y,
                    xref="x",
                    yref="paper",
                    text=aa,
                    showarrow=False,
                    font=dict(size=10, color=color),
                )
            )

    # Sequence all black
    else:
        for i, aa in enumerate(seq):
            annotations.append(
                dict(
                    x=i,
                    y=0.01,
                    xref="x",
                    yref="paper",
                    text=aa,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                )
            )

    fig.update_layout(annotations=annotations)
    return fig


def df_fasta_savs_to_2d_heatmap(df_fasta_savs, seq, pIRS_col="pIRS", norm=False):
    df_plot = df_fasta_savs.copy()
    assert (
        pIRS_col in df_plot.columns
    ), f"Specified pIRS column ({pIRS_col}) not found in DataFrame"

    df_plot["peptide_pos"] = df_plot["id"].apply(lambda x: x.split("___")[-2][1:-1])
    df_plot["mut"] = df_plot["id"].apply(lambda x: x.split("___")[-2])
    df_plot["mut_to"] = df_plot["id"].apply(lambda x: x.split("___")[-2][-1])

    # Aggregate on mut
    df_plot = (
        df_plot.groupby(["peptide_pos", "mut", "mut_to"])
        .agg({pIRS_col: "max"})
        .reset_index()
    )

    aa_order = list("CWGMPSTDLEFQNAHVYRIKX")
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    df_plot["mut_idx"] = df_plot["mut_to"].apply(lambda x: aa_to_idx[x])
    df_plot["peptide_pos"] = df_plot["peptide_pos"].astype(int)

    arr = np.full(shape=(len(seq), len(aa_order)), fill_value=np.nan)
    for i in range(len(df_plot)):
        row = df_plot.iloc[i]
        pos = row["peptide_pos"] - 1
        pos_aa = aa_to_idx[row["mut_to"]]
        arr[pos, pos_aa] = row[pIRS_col]

    # Normalize by wt_score?
    # if norm:
    #     for pos, aa_orig in enumerate(seq):
    #         wt_score = arr[pos, aa_to_idx[aa_orig]]
    #         arr[pos] = arr[pos] - wt_score

    # Annotate w/ sequence and cols
    aa_order = list("CWGMPSTDLEFQNAHVYRIKX")
    df_plot = pd.DataFrame(arr)
    df_plot.columns = aa_order
    df_plot.index = list(seq)

    return df_plot


def process_df_fasta(df_in):
    df_fasta = df_in.copy()
    # Rename columns
    df_fasta = df_fasta.rename(
        {"peptide_seq": "peptide", "peptide_pos": "Position", "pIRS": "pIRS"}, axis=1
    )

    all_sequences = pd.DataFrame()
    for name in df_fasta["id"].unique():
        df_seq = df_fasta[df_fasta["id"] == name].copy()
        all_sequences = pd.concat([all_sequences, df_seq])
    # Order
    all_sequences["aa"] = all_sequences["peptide"].apply(lambda s: s[0])
    all_sequences["aa_pos"] = all_sequences["aa"] + all_sequences["Position"].astype(
        str
    )
    all_sequences["Position_end"] = all_sequences["Position"] + 14
    # Assign lowest peptide for this position
    lowest_pos = all_sequences.groupby("Position")["pIRS"].idxmin()
    for pos, _id in zip(lowest_pos.index, lowest_pos.values):
        mask = all_sequences["Position"] == pos
        lowest_peptide = all_sequences.loc[mask].loc[_id, "peptide"]
        all_sequences.loc[mask, "lowest_peptide"] = lowest_peptide

    # Get core positions - OPTIMIZED VERSION
    if "core_pos" in all_sequences.columns:
        # Define formatting functions

        def color_red_bold(s):
            return f"<b><span style='color:red'>{s}</span></b>"

        def color_red(s):
            return f"<span style='color:red'>{s}</span>"

        def color_red_underline(s):
            return f"<span style='color:red'>{s}</span>"


        def color_green(s):
            return f"<span style='color:green'>{s}</span>"

        def underline(s):
            return f"<u>{s}</u>"

        def nothing(s):
            return s

        # Apply peptide formatting - this is more complex due to string slicing
        def format_peptide_90_plus(row):
            peptide = row["peptide"]
            core_pos = row["core_pos"]
            return f"{peptide[:core_pos]}{color_red_bold(peptide[core_pos:core_pos+9])}{peptide[core_pos+9:]}"

        def format_peptide_83_plus(row):
            peptide = row["peptide"]
            core_pos = row["core_pos"]
            return f"{peptide[:core_pos]}{color_red_underline(peptide[core_pos:core_pos+9])}{peptide[core_pos+9:]}"

        def format_peptide_below_83(row):
            peptide = row["peptide"]
            core_pos = row["core_pos"]
            return f"{peptide[:core_pos]}{underline(peptide[core_pos:core_pos+9])}{peptide[core_pos+9:]}"

        def format_peptide_ref(row):
            peptide = row["peptide"]
            core_pos = row["core_pos"]
            return f"{peptide[:core_pos]}{color_green(peptide[core_pos:core_pos+9])}{peptide[core_pos+9:]}"

        # Create base pIRS_text column
        all_sequences["pIRS_text"] = all_sequences["pIRS_rank"].apply(
            lambda x: f"{x:.1f}%"
        )

        # Create masks for different conditions
        mask_90_plus = all_sequences["pIRS_rank"] >= 90
        mask_83_plus = (all_sequences["pIRS_rank"] >= 83) & (
            all_sequences["pIRS_rank"] < 90
        )
        mask_below_83 = all_sequences["pIRS_rank"] < 83

        # ref mask
        mask_ref = all_sequences["in_reference"] == True

        all_sequences0 = all_sequences.copy()

        # Apply pIRS_text formatting based on masks
        all_sequences.loc[mask_90_plus, "pIRS_text"] = all_sequences0.loc[
            mask_90_plus, "pIRS_text"
        ].apply(color_red_bold)
        all_sequences.loc[mask_83_plus, "pIRS_text"] = all_sequences0.loc[
            mask_83_plus, "pIRS_text"
        ].apply(color_red)
        all_sequences.loc[mask_ref, "pIRS_text"] = all_sequences0.loc[
            mask_ref, "pIRS_text"
        ].apply(nothing)  

        # Apply core_seq formatting
        all_sequences.loc[mask_90_plus, "core_seq"] = all_sequences0.loc[
            mask_90_plus, "core_seq"
        ].apply(color_red_bold)
        all_sequences.loc[mask_83_plus, "core_seq"] = all_sequences0.loc[
            mask_83_plus, "core_seq"
        ].apply(color_red)
        all_sequences.loc[mask_ref, "core_seq"] = all_sequences0.loc[
            mask_ref, "core_seq"
        ].apply(color_green)


        # Apply peptide formatting using vectorized operations
        all_sequences.loc[mask_90_plus, "peptide"] = all_sequences0.loc[
            mask_90_plus
        ].apply(format_peptide_90_plus, axis=1)
        all_sequences.loc[mask_83_plus, "peptide"] = all_sequences0.loc[
            mask_83_plus
        ].apply(format_peptide_83_plus, axis=1)
        all_sequences.loc[mask_below_83, "peptide"] = all_sequences0.loc[
            mask_below_83
        ].apply(format_peptide_below_83, axis=1)
        all_sequences.loc[mask_ref, "peptide"] = all_sequences0.loc[
            mask_ref
        ].apply(format_peptide_ref, axis=1)

        # Core marker - vectorized
        def create_core_marker(core_pos):
            spacing = " " * int(core_pos / 2.5)
            core_mid = " " * 14 + "^"
            return f"    {spacing}{core_mid}"

        all_sequences["core_marker"] = all_sequences["core_pos"].apply(
            create_core_marker
        )

    return all_sequences


def _process_plot_df_fasta_all_sequences(df_fasta, y_min=-0.125, y_max=0.11):

    # Adds pirs text
    all_sequences = process_df_fasta(df_fasta)

    # Cap lowest values
    all_sequences["pIRS"] = all_sequences["pIRS"].clip(lower=y_min)

    def color_red(s):
        return f"<span style='color:red'>{s}</span>"

    def color_red_bold(s):
        return f"<b><span style='color:red'>{s}</span></b>"

    def bold(s):
        return f"<b>{s}</b>"

    def color_red_bold_2(s):
        # characters 3 and 5 are black, remaining red
        # return f"<b><span style='color:red'>{s}</span></b>"
        # s = color_red(s[0:2]) + s[2:3] + color_red(s[3:4]) + s[4] + color_red(s[5:])
        s = color_red(s)
        s = underline(s)
        return s

    def underline(s):
        return f"<u>{s}</u>"

    def color_green(s):
        return f"<span style='color:green'>{s}</span>"

    # Add reference symbols
    all_sequences["in_reference_str"] = ""

    for gene in ["DRB1"]:
        col = f"in_reference"

        if col in all_sequences.columns:
            m = all_sequences[col] == True
            all_sequences[col] = ""
            all_sequences.loc[m, col] = " )"
            all_sequences.loc[~m, col] = ")"

            col_core_seq = f"core_seq"
            S_core_seqs = all_sequences[col_core_seq].copy()

            # Replace I with 1
            S_core_seqs = S_core_seqs.str.replace("I", "Ⅰ")

            # In reference
            all_sequences.loc[m, col_core_seq] = S_core_seqs.loc[m].apply(
                lambda s: (
                    f"<span style='color:green'>{s}</span>" if isinstance(s, str) else s
                )
            )
            all_sequences.loc[m, "in_reference_str"] += f"{color_green(gene)} "

    # Add references
    m = all_sequences["in_reference_str"].apply(lambda s: len(s) == 0)
    all_sequences.loc[m, "in_reference_str"] = "❌"
    # all_sequences.loc[m, "in_reference_str"] = "✅"

    n_seqs = len(df_fasta["id"].unique())
    if n_seqs <= 10:
        # Use 'Set1' color palette for less than 10 sequences
        custom_colors = px.colors.qualitative.Dark24[:n_seqs]
    else:

        def generate_distinct_colors(n, saturation=0.5, value=0.9):
            HSV_tuples = [((1 - x / n) * 0.8, saturation, value) for x in range(n)]
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_list = [
                "rgb" + str(tuple(map(lambda x: int(255 * x), rgb)))
                for rgb in RGB_tuples
            ]
            color_list[-1] = "black"  # Set first color to black
            return color_list

        custom_colors = generate_distinct_colors(n_seqs)

    # Reverse
    # all_sequences = all_sequences.iloc[::-1]

    desired_order = all_sequences["id"].unique()

    return all_sequences, desired_order, n_seqs, custom_colors


def _get_gene_order_strs2(df_fasta):
    def color_red_bold(s):
        return f"<b><span style='color:red'>{s}</span></b>"

    def color_red(s):
        return f"<span style='color:red'>{s}</span>"

    def color_green(s):
        return f"<span style='color:green'>{s}</span>"

    # Pre-compute gene columns once
    gene_cols = [
        gene for gene in ["DRB1"] if f"core_seq" in df_fasta.columns
    ]

    if not gene_cols:
        return [""] * len(df_fasta)

    # Pre-compute spacing mapping
    spacing_map = {
        "DRB1": " ",
        }

    # Extract all needed columns at once using vectorized operations
    pirs_cols = [f"pIRS_rank" for gene in gene_cols]
    core_cols = [f"core_seq" for gene in gene_cols]
    ref_cols = [f"in_reference" for gene in gene_cols]

    # Get numpy arrays for faster access
    pirs_data = df_fasta[pirs_cols].values
    core_data = df_fasta[core_cols].values
    ref_data = df_fasta[ref_cols].values == True

    order_strings = []

    for i in range(len(df_fasta)):
        # Get data for this row
        pirs_values = pirs_data[i]
        core_values = core_data[i]
        ref_values = ref_data[i]

        # Filter out NaN values and get valid indices
        valid_mask = ~pd.isna(pirs_values)
        if not valid_mask.any():
            order_strings.append("")
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_pirs = pirs_values[valid_indices]

        # Sort by pIRS rank (descending)
        sorted_indices = valid_indices[np.argsort(valid_pirs)[::-1]]

        # Build string parts
        parts = []
        for j in sorted_indices:
            gene = gene_cols[j]
            pirs_rank = pirs_values[j]
            core_seq = core_values[j]
            in_reference = ref_values[j]

            # Apply coloring
            if in_reference:
                core_seq = color_green(core_seq)

            elif pirs_rank >= 90:
                core_seq = color_red_bold(core_seq)
            elif pirs_rank >= 83:
                core_seq = color_red(core_seq)

            spacing = spacing_map[gene]
            parts.append(f"<br>     {gene}:{spacing}{pirs_rank:.1f}% ({core_seq}) ")

        order_strings.append("".join(parts).strip())

    return order_strings


def plot_df_fasta_all_sequences_merged(
    # df_fasta, cmap=None, percentiles_85_95_99=[0.0384, 0.232, 0.609], y_min=-0.125, y_max=0.11, width=1250, height=500
    df_fasta,
    percentiles_85_95_99=(),
    gene_class="",
    y_min=-0.125,
    y_max=1.1,
    width=1250,
    height=500,
):

    if len(percentiles_85_95_99) != 3:

        try:
            # print(f"Trying to load percentiles from {model_dir}/quantilenormalizer.pkl")
            qt = joblib.load(f"model/DRB1/peptide/quantilenormalizer.pkl")
            values = [[0.85], [0.95], [0.99]]
            q85, q95, q99 = qt.inverse_transform(values).reshape(-1)
            percentiles_85_95_99 = np.round([q85, q95, q99], 3)
            print(f"pIRS percentiles 85%, 95%, 99% are {percentiles_85_95_99}")

        except Exception as e:
            percentiles_85_95_99 = [0.088903, 0.285847, 1.017301]
            print(
                f"pIRS percentiles not provided. Assigning 85%, 95%, 99% as {percentiles_85_95_99}"
            )

    # Reverse
    df_fasta = df_fasta.iloc[::-1].copy()
    all_sequences, desired_order, n_seqs, custom_colors = (
        _process_plot_df_fasta_all_sequences(df_fasta, y_min, y_max)
    )

    # # Get gene order strings
    # start = time.time()
    # order_strs = _get_gene_order_strs(df_fasta)
    # all_sequences["gene_order_str"] = order_strs

    # Get gene order strings
    order_strs = _get_gene_order_strs2(df_fasta)
    all_sequences["gene_order_str"] = order_strs

    title = f"Predicted peptide immunogenicity across {n_seqs} sequences"
    if gene_class:
        title = f"{gene_class} " + title

    # all_sequences["pIRS_text"] = all_sequences["pIRS_rank"].apply(lambda x: f"{x:.1f}%")

    hover_data = [
        "id",
        "peptide",
        "lowest_peptide",
        "Position_end",
        "pIRS_text",
        "pIRS_rank",
        "core_seq",
        "in_reference_str",
        #"DRB1_pIRS_rank",
        #"DRB1_in_reference",
        #"DRB1_core_seq",
        "core_marker",
        "gene_order_str",
    ]
    missing_cols = [col for col in hover_data if col not in all_sequences.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame for hover data: {missing_cols}")

    fig = px.line(
        all_sequences,
        x="Position",
        y="pIRS",
        color="id",
        category_orders={"id": desired_order},
        # category_orders={"id": max_IRS_order},
        title=title,
        hover_data=hover_data,
        custom_data=hover_data,
        color_discrete_sequence=custom_colors,  # Use the custom color palette
    )

    hovertemplate = """
    Name: %{customdata[0]}<br>
    Position: %{x} - %{customdata[3]}<br>
    <br>
    Predicted IRS: %{customdata[4]}<br>
    <br>
    Peptide: %{customdata[1]}<br>
    DRB1 core:  %{customdata[6]}<br>
    """
    # Reference
    hovertemplate += """
    <br>
    In human reference: %{customdata[7]}
    <extra></extra>
    """

    # Add color to markers black
    fig.update_traces(
        line=dict(width=2),
        mode="lines+markers",
        marker=dict(size=3, color="black"),
        hovertemplate=hovertemplate,
    )

    fig.update_xaxes(title_text="15-mer peptide starting position")


    fig.update_layout(
        hovermode="closest",
        hoverdistance=200,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font_color="black",
            bordercolor="gray",
        ),
        legend_title_text="Name",
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", x=1.045),
    )

    # X axis slider!
    fig.update_layout(
        dragmode="zoom",  # 'pan' or 'select' are other options
    )

    # Y axis with 2 digit precision
    fig.update_yaxes(tickformat=".2f")

    # Set y-limit max to either 0.07 or max value
    y_max = all_sequences["pIRS"].max() + 0.01
    y_min = 0
    # y_max = y_max2 if y_max2 > y_max else y_max
    # Add percentage y axis
    plotly_secondary_y_axis(
        fig,
        yticks=percentiles_85_95_99,
        ytick_labels=["85%", "95%", "99%"],
        y_min=y_min,
        y_max=y_max,
    )

    # Auto layout
    fig.update_layout(
        autosize=True,
        width=width,
        height=height,
    )

    n_pos = len(all_sequences["Position"].unique())
    if n_pos == 1:
        fig.update_xaxes(range=[0.80, 1.20])
        fig.update_xaxes(
            title_text="15-mer peptide starting position",
            tickvals=[0, 1],
            ticktext=["1"],
        )

    # When double clicking, always reset to this view
    # First trace mask from desired order
    m = all_sequences["id"] == desired_order[0]
    fig.update_layout(
        dragmode="zoom",
        hovermode="closest",
        hoverdistance=200,
        # spikedistance=400,
        xaxis=dict(
            range=[
                all_sequences.loc[m, "Position"].min() - 1,
                all_sequences.loc[m, "Position"].max() + 1,
            ]
        ),
        yaxis=dict(range=[y_min, y_max]),
        xaxis2=dict(
            range=[
                all_sequences.loc[m, "Position"].min() - 1,
                all_sequences.loc[m, "Position"].max() + 1,
            ]
        ),
        yaxis2=dict(range=[y_min, y_max]),
    )

    # Reverse legend order
    fig.update_layout(legend=dict(traceorder="reversed"))

    return fig


def process_df_pirs_to_arr(df_pirs, irs_col="pIRS", order=False):

    def get_arr_order(arr):

        arr2 = arr.copy()
        arr2[np.isnan(arr2)] = -1
        D = squareform(pdist(arr2, metric="euclidean"))

        # Start with the first vector and build the ordering using a nearest neighbor approach
        n = arr.shape[0]
        order = [0]
        remaining = list(range(1, n))

        while remaining:
            last = order[-1]
            # Find the index in 'remaining' closest to the last added vector
            next_index = min(remaining, key=lambda i: D[last, i])
            order.append(next_index)
            remaining.remove(next_index)

        return order

    # Determine dimensions from the data
    L = df_pirs["peptide_pos"].max()
    N = df_pirs["id"].nunique()

    # Create arrays with default values
    arr_seq = np.full((N, L), "X" * 20)
    arr_in_ref = np.full((N, L), "False")
    arr = np.full((N, L), np.nan)

    # Get unique ids and fill the arrays
    ids = df_pirs["id"].unique()
    for i, id_val in enumerate(ids):
        df = df_pirs[df_pirs["id"] == id_val]
        arr[i, df["peptide_pos"].values - 1] = df[irs_col].values

        # Peptide space every 5 amino acids
        pep_seq = df["peptide_seq"].apply(
            lambda s: s[0:5] + "-" + s[5:10] + "-" + s[10:15]
        )
        arr_seq[i, df["peptide_pos"].values - 1] = pep_seq
        arr_in_ref[i, df["peptide_pos"].values - 1] = (
            df["in_reference"].apply(lambda s: "True" if s else "False").values
        )

    if order:
        # Get the order of sequences based on the immunogenicity data
        print("Ordering sequences based on immunogenicity data ...")
        order = get_arr_order(arr)
        arr = arr[order]
        arr_seq = arr_seq[order]
        ids = ids[order]
        arr_in_ref = arr_in_ref[order]

    # Prep plot
    df_plot = pd.DataFrame(arr, index=range(1, len(arr) + 1), columns=range(1, L + 1))
    df_plot_seq = pd.DataFrame(
        arr_seq, index=range(1, len(arr_seq) + 1), columns=range(1, L + 1)
    )
    df_plot_in_ref = pd.DataFrame(
        arr_in_ref, index=range(1, len(arr_in_ref) + 1), columns=range(1, L + 1)
    )

    return df_plot, ids, df_plot_seq, df_plot_in_ref
