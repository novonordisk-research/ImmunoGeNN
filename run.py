import argparse
import os
import sys
import time

import biolib
import biolib._internal.utils.multinode as biolib_multinode
from biolib.utils import SeqUtil
from tqdm import tqdm
import shutil

import src.mutation_editor
import src.plots_pred
import src.utils
import src.deimm
import copy
import os
import numpy as np

def parse_args():

    def _boolean_string(string):
        if string.lower() in ("true", "1"):
            return True
        elif string.lower() in ("false", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Run ImmunoGeNN predictions")
    parser.add_argument("--mode", default="", type=str, help="Compare")

    parser.add_argument(
        "--model_names_str",
        type=str,
        default="DRB1",
        help="Path to the input FASTA file",
    )


    parser.add_argument(
        "--fasta_file",
        type=str,
        default="data/input.fasta",
        help="Path to the input FASTA file",
    )
    parser.add_argument(
        "--human_references_pkl",
        type=str,
        default="data_record/human_references_9mers.pkl.lz4",
        #default="false",
        help="Path human references pickle file",
    )
    parser.add_argument(
        "--extra_references",
        type=str,
        default="",
        help="Path to extra references file",
    )
    parser.add_argument(
        "--skip_plots",
        type=_boolean_string,
        default="false",
        help="Skips plots if set to true",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.040, help="Threshold for plotting"
    )
    parser.add_argument(
        "--deimmunize_first_sequence",
        type=_boolean_string,
        default="false",
        help="Deimmunize sequence plot",
    )
    parser.add_argument(
        "--variants_to_generate",
        default=10,
        type=int,
        help="Deimmunizing variants to generate",
    )
    parser.add_argument(
        "--ranges_str",
        default="",
        help="Ranges to deimmunize, e.g. 1-100,200-300. Default empty for all",
    )
    parser.add_argument(
        "--plot_first_n", type=int, default=1, help="Number of sequences to plot"
    )
    parser.add_argument("--top_n", type=int, default=20, help="ESM variants to show")
    parser.add_argument(
        "--esm_model", type=str, default="esm2_t6_8M_UR50D", help="ESM model to use"
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        default="data/_sequences/lyzl4/input.fasta",
        help="Path to the input FASTA file",
    )
    parser.add_argument(
        "--outdir", type=str, default="output", help="Output directory for saving plots"
    )
    parser.add_argument(
        "--verbose", default=0, type=int, help="Increase output verbosity"
    )
    parser.add_argument("--advanced", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # # Model list
    # args.model_list = args.models_str.split(",")

    model_list = args.model_names_str.split(",")
    args.model_list = [f"model/{name}/" for name in model_list]

    # Human refs
    if args.human_references_pkl == "false":
        args.human_references_pkl = ""

    # Check models exists
    for model_path in args.model_list:
        pep_model = f"{model_path}/peptide/model.pkl"
        core_model = f"{model_path}/core/model.pkl"
        assert os.path.exists(model_path), f"Unable to locate model dir: {model_path}"
        assert os.path.exists(pep_model), f"Unable to locate model: {pep_model}"
        assert os.path.exists(core_model), f"Unable to locate model: {core_model}"

    # Check ranges
    # Per region
    if args.ranges_str == "":
        args.ranges = [(1, np.inf)]
    else:
        args.ranges_str
        args.ranges = [tuple(map(int, r.split("-"))) for r in args.ranges_str.split(",")]
        print(f"Using ranges: {args.ranges}")

    return args


def return_df_seq_or_false(df_fasta, i):
    names = df_fasta["id"].unique()

    if i < 0 or i >= len(names):
        return False

    df_seq = df_fasta[df_fasta["id"] == names[i]]

    return df_seq




def parse_fasta_records(fasta_file):

    alphabet = list("ACDEFGHIKLMNPQRSTVWYX")

    # Parse records for shorter id, invalid sequences
    records = list(SeqUtil.parse_fasta(fasta_file))
    for i, record in enumerate(records):

        record.id = str(record.id).split(" ")[0][:100]

        if len(record.sequence) < 15:
            print(f"Warning: Removing too short sequence: {record.id}")
            records.remove(record)

        elif not all([c in alphabet for c in record.sequence]):
            invalid = [c for c in record.sequence if c not in alphabet][:10]
            print(
                f"Warning: record {i+1}/{len(records)} ({record.id}) contains {len(invalid)} invalid residue(s): {', '.join(invalid)} ..."
            )
            records.remove(record)

    assert len(records) > 0, "No valid sequences found in fasta file"

    return records


def plotly_fig_to_html(fig, outname, include_plotlyjs=False):
    """Inject local plotly source into plotly html file"""
    fig.write_html(
        outname,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={"doubleClick": "reset"},
    )


def plot_all_sequences_merged(
    df_fasta, gene_class="", outdir="output", include_plotlyjs=True
):
    fig = src.plots_pred.plot_df_fasta_all_sequences_merged(
        df_fasta, gene_class=gene_class, width=1100, height=475
    )

    plotly_fig_to_html(fig, f"{outdir}/plots.html", include_plotlyjs=include_plotlyjs)
    return fig

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


def print_time(start_time, name, verbose=True):
    if verbose >= 2:
        print(f"{name}: {time.time() - start_time:.2f}")


def main(args):

    # Load references
    human_refs_set = set()
    if args.human_references_pkl:
        start = time.time()
        human_refs_set = src.utils.load_references(
            human_references_pkl=args.human_references_pkl,
            extra_references=args.extra_references,
            verbose=args.verbose,
        )
        print(
            f"\nLoaded {len(human_refs_set):,} peptides from human reference (131 MB) in {time.time()-start:.3f} seconds\n"
        )

    # Parse records
    records = parse_fasta_records(args.fasta_file)

    # Collect
    SAV_files = []
    pIRS_files = []

    # Option 1) Deimmunize first sequence
    if args.deimmunize_first_sequence:

        with tqdm(
            total=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:

            record = records[0]
            N_variants = (len(record.sequence) - 14) * 20 * 15
            # Thousand separator
            pbar.set_description(
                f"Screening {N_variants:,} possible deimmunizing mutations in {record.id} ..."
            )
            pbar_percent(pbar, 33)
            pbar.update()
            sys.stderr.flush()

            # Flush print

            # Predict SAVs
            start = time.time()
            df_fasta_savs = src.utils.seq_to_df_fasta_savs(record.sequence, record.id)
            pbar_percent(pbar, 50)

            start = time.time()
            _, df_outname_list = src.utils.predict_df_fasta(
                model_list=args.model_list,
                df_fasta=df_fasta_savs,
                human_refs_set=human_refs_set,
                outname="SAVs.csv",
                outdir=args.outdir,
                save=True,
            )

            SAV_files.extend(df_outname_list)
            pbar_percent(pbar, 100)
            pbar.close()

    # Option 2) Predict all
    else:

        batches = biolib_multinode.fasta_batch_records(
            records,
            work_per_batch_min=2e6,  # Max 2M residues per batch
            verbose=False,
        )

        # Otherwise
        with tqdm(
            total=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:

            # df_fasta_list = []
            for batch_n, batch_records in enumerate(batches):

                print("\n")
                pbar.set_description(
                    f"Predicting batch {batch_n+1}/{len(batches)}: {len(batch_records)} sequences"
                )
                print()
                time.sleep(0.01)
                pbar_percent(pbar, (batch_n + 0.5) / len(batches) * 100)
                time.sleep(0.01)

                # Predict and save pIRS + scores.csv
                df_fasta = src.processing.parse_records_to_15mer_df(batch_records)
                _, df_outname_list = src.utils.predict_df_fasta(
                    model_list=args.model_list,
                    df_fasta=df_fasta,
                    human_refs_set=human_refs_set,
                    outname=f"pIRS_batch__{batch_n+1}.csv",
                    outdir=args.outdir,
                    save=True,
                )

                pIRS_files.extend(df_outname_list)
                pbar_percent(pbar, (batch_n + 1) / len(batches) * 100)

            # Final progress bar
            print("\n")
            pbar.set_description("Plotting sequences ...")
            pbar_percent(pbar, 99)
            pbar.update()
            pbar.close()
            print()

            # Merge
            pIRS_files = src.utils.create_single_gene_pIRS_files(
                pIRS_files, args.outdir, remove=True
            )

    if len(pIRS_files) >= 1:

        # Merge pIRS files
        df_merged = src.utils.prepare_output_pirs_file(pIRS_files[0])

        # Cleanup DRB1 dir
        if os.path.exists(f"{args.outdir}/DRB1/"):
            shutil.rmtree(f"{args.outdir}/DRB1/")

        # Write
        src.utils.save_pirs_file_rounded(
            df_merged, f"{args.outdir}/pIRS.csv"
        )
        print(f"Writing merged pIRS to {args.outdir}/pIRS.csv")

        # Write scores.csv
        # start = time.time()
        src.utils.save_df_fasta_to_pIRS_scores_csv(df_merged, args.outdir, save=True)

        if args.skip_plots:
            print("\nSkipping plots as requested (--skip_plots true)\n")
            return

        # Only plot first 100 sequences
        IDS = df_merged["id"].unique()[:100]
        df_merged = df_merged[df_merged["id"].isin(IDS)]

        counts = df_merged["id"].value_counts()
        keep = counts[counts < 1000].index.to_list()
        df_merged = df_merged[df_merged["id"].isin(keep)]

        # Skip sequences >= 1000 residues
        if len(df_merged) >= 1:
            # Plot merged
            plot_all_sequences_merged(
                df_merged,
                gene_class="",
                outdir=args.outdir,
                include_plotlyjs=True,
            )
            print(f"Writing merged plots to {args.outdir}/plots.html")

        else:
            print(f"No sequences <= 1000 residues to plot (of {len(IDS)} total)")

    # Plot SAVs if any
    if SAV_files:

        df_merged = src.utils.merge_sav_files(SAV_files, verbose=args.verbose)

        df_out = df_merged.copy()
        df_out["pIRS"] = df_out["pIRS"].apply(lambda x: f"{x:.5f}")
        df_out["pIRS_rank"] = df_out["pIRS_rank"].apply(lambda x: f"{x:.3f}")
        df_out.to_csv(f"{args.outdir}/SAVs.csv", index=False)

        # Prepare top n deimmunizing variants
        record_orig = copy.deepcopy(record)

        # Rank deimmunizing variants
        df_mut = src.deimm.df_fasta_savs_to_df_mut(df_merged, record_orig)
        df_mut_weighted = src.deimm.weight_df_mut(df_mut)

        # Select n per region
        all_records = []
        for r in args.ranges:
            start, end = r
            m = (df_mut_weighted['pos'] >= start) & (df_mut_weighted['pos'] <= end)
            df_sub = df_mut_weighted[m]

            print(f"\nDeimmunizing variants in range: {start}-{end} ({df_sub.shape[0]} available mutations)")
            records = src.deimm.df_mut_weighted_to_fasta_records(df_sub, record, top_n=10)
            all_records.extend(records)
            #records = src.deimm.df_mut_weighted_to_fasta_records(df_mut_weighted, record_orig, top_n=20)

        outfile = f"{args.outdir}/deimmunized_variants.fasta"
        biolib.utils.SeqUtil.write_records_to_fasta(outfile, all_records)
        print(f"Wrote {len(records)-1} deimmunizing variant sequences to {outfile}")

        if args.skip_plots:
            print("\nDone! Skipping plots as requested (--skip_plots true)\n")
            return

        # Re-predict
        records = parse_fasta_records(outfile)
        # Predict and save pIRS + scores.csv
        df_fasta = src.processing.parse_records_to_15mer_df(records)
        _, pIRS_files = src.utils.predict_df_fasta(
            model_list=args.model_list,
            df_fasta=df_fasta,
            human_refs_set=human_refs_set,
            outname=f"deimm.csv",
            outdir=args.outdir,
            save=True,
        )

        df_merged2 = src.utils.merge_all_gene_pIRS_files(pIRS_files)
        plot_all_sequences_merged(
            df_merged2,
            gene_class="",
            outdir=args.outdir,
            include_plotlyjs=True,
        )
        print(f"Writing merged plots to {args.outdir}/plots.html")

        # Set pIRS 0 to pIRS_rank 0
        m = ~(df_merged["pIRS"] > 0)
        df_merged.loc[m, "pIRS_rank"] = 0

        # Add clickable mutation editor
        fig = src.plots_pred.plot_deimmunization_plot(
            df_merged,
            record,
            top_n=args.top_n,
            gene_class="",
            save=False,
            width=1100,
            height=475,
        )
        # plotly_fig_to_html(fig, f"{args.outdir}/SAVs.html", include_plotlyjs=True)
        # print(f"Writing merged SAVs plots to {args.outdir}/SAVs.html")
        src.mutation_editor.add_fig_clickable_mutation_editor(fig, record, args.outdir)
        merge_files = [
            f"{args.outdir}/plots.html",
            f"{args.outdir}/mutation_editor.html",
        ]
        src.utils.merge_html_files(
            merge_files=merge_files,
            output_file=f"{args.outdir}/plots.html",
            remove=False,
            verbose=args.verbose,
        )
        #os.remove(f"{args.outdir}/mutation_editor.html")

        print(f"\nPlots saved to {args.outdir}/plots.html")


def pbar_percent(pbar, target_percentage):
    """
    Set the progress bar to an artificial percentage.

    :param pbar: tqdm progress bar object
    :param target_percentage: desired percentage (0-100)
    """
    current_percentage = pbar.n
    if target_percentage > current_percentage:
        difference = target_percentage - current_percentage
        pbar.update(difference)

    elif target_percentage < current_percentage:
        pbar.n = target_percentage

    pbar.refresh()


if __name__ == "__main__":

    pre_main = time.time()

    # Make output directory and plotly source
    args = parse_args()

    # Set BioLib job name (if running on BioLib platform)
    biolib.sdk.Runtime.set_result_name_from_file(args.fasta_file)

    # Run main
    main(args)
