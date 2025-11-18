import os

import joblib
import numpy as np
import pandas as pd
from biolib.utils import SeqUtil

import src.utils


def df_fasta_to_test_time_aligned_seqs(df_fasta, core_pos, peptide_col="peptide_seq"):

    if not peptide_col in df_fasta.columns:
        raise Exception("Missing column peptide_seq")
    # df_fasta["peptide_seq_actual"] = df_fasta["peptide_seq"]
    # assert "peptide_seq_actual" in df_fasta.columns, "Missing column peptide_seq_actual"

    # Prepare to align peptides in array
    arr_peptides = np.array([list(s) for s in df_fasta[peptide_col].values])

    # X array
    arr = np.full(shape=(len(df_fasta), 21), fill_value="X")

    # Align 9-mer binding cores in 15-mer peptides to array
    # Such that 9-mer binding cores always start from 6th position (15-9)
    # And peptides may stretch from 6 to 21st position (6+15)

    # Ignore outlier core pos (inverted)
    if core_pos < 0 or core_pos > 6:
        assert False, f"Invalid core_pos: {core_pos}"

    # mask = df_fasta["core_freq_pos"] == core_pos
    mask = np.ones(len(df_fasta), dtype=bool)

    start = 6 - core_pos
    end = 6 + 15 - core_pos
    arr[mask, start:end] = arr_peptides[mask]

    aligned_seqs = ["".join(L) for L in arr]

    return aligned_seqs


def fasta_to_df_fasta(fasta_file):
    records = list(SeqUtil.parse_fasta(fasta_file))
    print(f"Read {len(records)} FASTA sequences from {fasta_file}")

    fasta_dict = {}
    for record in records:
        fasta_dict[record.id] = {
            "peptide_seq_actual": record.sequence,
        }

    df_fasta = pd.DataFrame.from_dict(
        fasta_dict, orient="index", columns=["peptide_seq", "peptide_seq_actual"]
    )
    df_fasta.insert(0, "id", df_fasta.index)

    return df_fasta


def transform(normalizer, scores):
    """Transform scores using the provided normalizer."""
    scores_norm = normalizer.transform(scores.reshape(-1, 1))
    return scores_norm.reshape(-1)


def inverse_transform(normalizer, scores):
    """Transform scores using the provided normalizer."""
    scores_norm = normalizer.inverse_transform(scores.reshape(-1, 1))
    return scores_norm.reshape(-1)


def normalize_y_hat_to_0_100(y_hat, model_dir, qt_train_pred_path="", verbose=True):
    """
    Transforms y_hat range (-2% to >120%) to IRS% range (0% to 100%).

    df_train = pd.read_csv("notebooks/data/predictions_train.csv")
    qt_train_pred = QuantileTransformer(output_distribution='uniform', random_state=42)
    qt_train_pred.fit(df_train["y_pred"].values.reshape(-1, 1))

    # Save
    import joblib
    joblib.dump(qt_train_pred, "model/qt_train_pred_latest.pkl")
    """

    if not qt_train_pred_path:
        qt_train_pred_path = f"{model_dir}/qt_train_pred_latest.pkl"

    # Normalizer fitted on training set predicted pIRS
    assert os.path.exists(qt_train_pred_path), f"Missing {qt_train_pred_path}"
    if verbose:
        print(f"Loading qt_train_pred from {qt_train_pred_path}")

    qt_train_pred = joblib.load(qt_train_pred_path)
    pirs_rank = qt_train_pred.transform(y_hat.reshape(-1, 1)).reshape(-1)

    return pirs_rank * 100


def normalize_y_hat_0_100_to_irs(irs_rank, model_dir, qt="", verbose=True):
    """
    Transforms pIRS% (0% to 100%) to pIRS range (0 to inf)
    qt_train_true_path=model/qt_train_true_latest.pkl
    """

    if not qt:
        # qt_train_true_path = f"{model_dir}/qt_train_true_latest.pkl"
        qt = f"{model_dir}/quantilenormalizer.pkl"

    # IRS% is back-transformed to IRS (0 to inf)
    assert os.path.exists(qt), f"Missing {qt}"
    if verbose:
        print(f"Loading qt from {qt}. Expecting range 0.0 - 100.0")
    qt = joblib.load(qt)

    irs = inverse_transform(qt, irs_rank / 100)  # Hack to 0-1 range

    return irs


def normalize_irs_to_irs_rank(
    pirs_rank, model_dir, qt_train_true_path="", verbose=True
):
    """
    Transforms pIRS% (0% to 100%) to pIRS range (0 to inf)
    qt_train_true_path=model/qt_train_true_latest.pkl
    """

    if not qt_train_true_path:
        qt_train_true_path = f"{model_dir}/qt_train_true_latest.pkl"
        # qt_train_true_path = f"{model_dir}/quantilenormalizer.pkl"

    # IRS% is back-transformed to IRS (0 to inf)
    assert os.path.exists(qt_train_true_path), f"Missing {qt_train_true_path}"
    if verbose:
        print(f"Loading qt_train_true from {qt_train_true_path}")
    qt_train_true = joblib.load(qt_train_true_path)
    pirs = inverse_transform(qt_train_true, pirs_rank)

    return pirs


def normalize_y_hat_to_pirs(
    y_hat, model_dir, qt_train_pred_path="", qt_train_true_path="", verbose=True
):
    """
    Transforms y_hat (range -2% to >120%) to pIRS range (0 to inf)
    """

    if not qt_train_pred_path or not qt_train_true_path:
        qt_train_pred_path = f"{model_dir}/qt_train_pred_latest.pkl"
        qt_train_true_path = f"{model_dir}/qt_train_true_latest.pkl"
        qt_train_true_path = f"{model_dir}/quantilenormalizer.pkl"

    y_hat_0_100 = normalize_y_hat_to_0_100(
        y_hat, model_dir=model_dir, qt_train_pred_path=qt_train_pred_path, verbose=False
    )
    pirs = normalize_y_hat_0_100_to_irs(y_hat_0_100, model_dir=model_dir, verbose=False)

    return pirs


def preprocess_df_fasta_binding_cores(df_fasta, peptide_col="peptide_seq"):

    # Need to give unique index for steps to work
    indices = (df_fasta["id"] + np.arange(len(df_fasta)).astype(str)).values

    # Find max scoring 15mer by aligning all possible 9mer cores
    L = []
    for core_pos in [0, 1, 2, 3, 4, 5, 6]:
        aligned_seqs = df_fasta_to_test_time_aligned_seqs(df_fasta, core_pos, peptide_col=peptide_col)
        S = pd.Series(aligned_seqs)
        S.name = peptide_col
        S.index = indices
        L.append(S)

    df_try = pd.DataFrame(pd.concat(L))
    df_try.insert(0, "id", df_try.index)
    df_try["irs"] = np.nan
    df_try = df_try.loc[indices]

    # get df_try core_seq
    # df_try["core_seq"] = df_try["peptide_seq"].apply(lambda s: s[6:6+9])
    X_input, _ = data_to_onehot_X_Y(df_try, validate=False, peptide_col=peptide_col)

    return X_input, df_try


def get_df_try_cores(df_try, df_fasta):
    df_try_top = (
        df_try.sort_values("pIRS_rank", ascending=False)
        .groupby(level=0)
        .apply(lambda x: x.head(3))
    )
    df_try_top["core_pred"] = df_try_top["peptide_seq"].apply(lambda s: s[6:15])
    top_3_core_pred = df_try_top.loc[df_fasta.index, "core_pred"].values.reshape(-1, 3)
    return top_3_core_pred


def get_df_try_scores_rank(df_try):
    scores = (
        df_try.sort_values("pIRS")
        .groupby(level=0)
        .pIRS.apply(lambda x: x.tail(1).mean())
    )
    scores_rank = (
        df_try.sort_values("pIRS_rank")
        .groupby(level=0)
        .pIRS_rank.apply(lambda x: x.tail(1).mean())
    )

    return scores, scores_rank



def immunogenn_predict_df_fasta_trying_binding_cores3(
    model_path,
    model_dir,
    df_fasta,
    X_input,
    cap=False,
    normalize=True,
    peptide_col="peptide_seq",
):

    # Predict
    y_pred, y_pred_rank = src.utils.model_predict_pirs_and_rank(
        model_path, X_input, model_dir=model_dir, normalize=normalize, cap=cap
    )

    # Get cores (df_try always 7 cores)
    y_pred_cores = y_pred.reshape(-1, 7)
    y_pred_rank_cores = y_pred_rank.reshape(-1, 7)

    # Prep dataframe with pIRS, pIRS_rank and per-core IRS
    df_cores_irs = pd.DataFrame(y_pred_cores, index=df_fasta.index, columns=["0", "1", "2", "3", "4", "5", "6"])
    df_cores_irs.insert(0, "pIRS", np.max(y_pred_cores, axis=1))
    df_cores_irs.insert(1, "pIRS_rank", np.max(y_pred_rank_cores, axis=1))

    # add core_seq and pos
    core_pos = np.argmax(y_pred_rank_cores, axis=1)
    core_seqs = [pep[core_pos:core_pos+9] for pep, core_pos in zip(df_fasta[peptide_col].values, core_pos)]
    df_cores_irs.insert(2, "core_seq", core_seqs)
    df_cores_irs.insert(3, "core_pos", core_pos)

    # Add to df_fasta
    df_out = df_fasta.copy()
    update_cols = ["pIRS", "pIRS_rank", "core_seq", "core_pos", "0", "1", "2", "3", "4", "5", "6"]
    df_out[update_cols] = df_cores_irs[update_cols]
    #df_fasta = df_fasta.join(df_cores_irs)

    return df_out


def _get_core_pep_preds(
    model_dir,
    X_input,
    reuse_top_n_cores=3,
):

    pep_model_dir = f"{model_dir}/peptide"
    core_model_dir = f"{model_dir}/core"
    assert os.path.exists(pep_model_dir), f"Missing {pep_model_dir}"
    assert os.path.exists(core_model_dir), f"Missing {core_model_dir}"

    # 1) Predict with core model (all 7 cores)
    core_model_path = f"{core_model_dir}/model.pkl"
    core_y_hat, core_y_hat_rank = src.utils.model_predict_pirs_and_rank(
        core_model_path, X_input, model_dir=core_model_dir, 
    )

    # Reshape to get core distribution
    core_y_hat = core_y_hat.reshape(-1, 7)
    core_y_hat_rank = core_y_hat_rank.reshape(-1, 7)

    # 2) Predict with peptide model (only top 1ncore)
    n = reuse_top_n_cores
    core_max = np.argsort(core_y_hat_rank, axis=1)[:, -n:]
    core_max_idxs = (core_max + 7 * np.arange(len(core_max))[:, None]).flatten()

    # Re-use cores
    pep_model_path = f"{pep_model_dir}/model.pkl"
    pep_y_hat, pep_y_hat_rank = src.utils.model_predict_pirs_and_rank(
        pep_model_path, X_input[core_max_idxs], model_dir=pep_model_dir
    )

    # Get top preds
    pep_y_hat = np.max(pep_y_hat.reshape(-1, n), axis=1)
    pep_y_hat_rank = np.max(pep_y_hat_rank.reshape(-1, n), axis=1)

    return pep_y_hat, pep_y_hat_rank, core_y_hat, core_y_hat_rank

def predict_df_fasta_cores_and_pep(
    model_dir,
    df_fasta,
    X_input,
    peptide_col="peptide_seq",
):

    pep_y_hat, pep_y_hat_rank, core_y_hat, core_y_hat_rank = _get_core_pep_preds(
        model_dir,
        X_input,
        reuse_top_n_cores=7,
    )

    # Prep dataframe with pIRS, pIRS_rank and per-core IRS
    core_cols = ["0", "1", "2", "3", "4", "5", "6"]
    df_cores_irs = pd.DataFrame(core_y_hat, index=df_fasta.index, columns=core_cols)
    df_cores_irs.insert(0, "pIRS", pep_y_hat)
    df_cores_irs.insert(1, "pIRS_rank", pep_y_hat_rank)

    # Get core% distribution
    A = (df_cores_irs[core_cols] / df_cores_irs["pIRS"].values.reshape(-1, 1))
    # Top 3 only
    k = 3
    t = -np.partition(-A, k-1, axis=1)[:, k-1]
    A[A < t.reshape(-1, 1)] = 0
    # Percent
    A = np.round((A / A.sum(axis=1).values.reshape(-1, 1)) * 100, 3)
    df_cores_irs[core_cols] = A

    # add core_seq and pos
    core_pos = np.argmax(core_y_hat_rank, axis=1)
    core_seqs = np.array([pep[core_pos:core_pos+9] for pep, core_pos in zip(df_fasta[peptide_col].values, core_pos)])
    df_cores_irs.insert(2, "core_seq", core_seqs)
    df_cores_irs.insert(3, "core_pos", core_pos)
    df_cores_irs.insert(4, "peptide_seq", df_fasta[peptide_col].values)

    # Add to df_fasta
    df_out = df_fasta.copy()
    update_cols = ["pIRS", "pIRS_rank", "core_seq", "core_pos"] + core_cols
    df_out[update_cols] = df_cores_irs[update_cols]

    return df_out


def parse_records_to_15mer_df(records):

    # parse into 15mers into dict
    input_positions = []
    input_peptides = []
    for record in records:
        seq = record.sequence

        for i in range(len(seq) - 15 + 1):
            input_peptides.append((record.id, seq[i : i + 15]))

        input_positions.extend(range(1, len(seq) - 15 + 2))

    df = pd.DataFrame(input_peptides, columns=["id", "peptide_seq"])
    df.index = df["id"].values
    df.insert(1, "peptide_pos", input_positions)

    return df


def parse_fasta_to_15mer_df(fasta_file, n_records=-1):
    records = list(SeqUtil.parse_fasta(fasta_file))
    df = parse_records_to_15mer_df(records)
    
    return df


def data_to_onehot_X_Y(df, validate=True, peptide_col="peptide_seq"):
    assert peptide_col in df.columns, f"Missing column {peptide_col}"

    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    oh_dict = {aa: idx for idx, aa in enumerate(alphabet)}

    # Pre-allocate the output array
    max_len = df[peptide_col].str.len().max()
    n_samples = len(df)
    X_arr = np.zeros((n_samples, max_len * len(alphabet)), dtype=np.float32)

    for i, seq in enumerate(df[peptide_col]):
        for j, aa in enumerate(seq):
            X_arr[i, j * len(alphabet) + oh_dict[aa]] = 1

    if validate:
        assert "irs" in df.columns, "Missing column IRS"
        y_arr = df["irs"].values
    else:
        y_arr = None

    return X_arr, y_arr

