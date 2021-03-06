import typing
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.integrate as integrate


filename_to_cols = {
    "accepted_2007_to_2018Q4.csv": {
        "float_cols": [
            "member_id",
            "loan_amnt",
            "funded_amnt",
            "funded_amnt_inv",
            "int_rate",
            "installment",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
            "inq_last_6mths",
            "mths_since_last_delinq",
            "mths_since_last_record",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
            "out_prncp",
            "out_prncp_inv",
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_amnt",
            "last_fico_range_high",
            "last_fico_range_low",
            "collections_12_mths_ex_med",
            "mths_since_last_major_derog",
            "policy_code",
            "annual_inc_joint",
            "dti_joint",
            "acc_now_delinq",
            "tot_coll_amt",
            "tot_cur_bal",
            "open_acc_6m",
            "open_act_il",
            "open_il_12m",
            "open_il_24m",
            "mths_since_rcnt_il",
            "total_bal_il",
            "il_util",
            "open_rv_12m",
            "open_rv_24m",
            "max_bal_bc",
            "all_util",
            "total_rev_hi_lim",
            "inq_fi",
            "total_cu_tl",
            "inq_last_12m",
            "acc_open_past_24mths",
            "avg_cur_bal",
            "bc_open_to_buy",
            "bc_util",
            "chargeoff_within_12_mths",
            "delinq_amnt",
            "mo_sin_old_il_acct",
            "mo_sin_old_rev_tl_op",
            "mo_sin_rcnt_rev_tl_op",
            "mo_sin_rcnt_tl",
            "mort_acc",
            "mths_since_recent_bc",
            "mths_since_recent_bc_dlq",
            "mths_since_recent_inq",
            "mths_since_recent_revol_delinq",
            "num_accts_ever_120_pd",
            "num_actv_bc_tl",
            "num_actv_rev_tl",
            "num_bc_sats",
            "num_bc_tl",
            "num_il_tl",
            "num_op_rev_tl",
            "num_rev_accts",
            "num_rev_tl_bal_gt_0",
            "num_sats",
            "num_tl_120dpd_2m",
            "num_tl_30dpd",
            "num_tl_90g_dpd_24m",
            "num_tl_op_past_12m",
            "pct_tl_nvr_dlq",
            "percent_bc_gt_75",
            "pub_rec_bankruptcies",
            "tax_liens",
            "tot_hi_cred_lim",
            "total_bal_ex_mort",
            "total_bc_limit",
            "total_il_high_credit_limit",
            "revol_bal_joint",
            "sec_app_fico_range_low",
            "sec_app_fico_range_high",
            "sec_app_inq_last_6mths",
            "sec_app_mort_acc",
            "sec_app_open_acc",
            "sec_app_revol_util",
            "sec_app_open_act_il",
            "sec_app_num_rev_accts",
            "sec_app_chargeoff_within_12_mths",
            "sec_app_collections_12_mths_ex_med",
            "sec_app_mths_since_last_major_derog",
            "deferral_term",
            "hardship_amount",
            "hardship_length",
            "hardship_dpd",
            "orig_projected_additional_accrued_interest",
            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",
            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
        ],
        "object_cols": [
            "id",
            "term",
            "grade",
            "sub_grade",
            "emp_title",
            "emp_length",
            "home_ownership",
            "verification_status",
            "issue_d",
            "loan_status",
            "pymnt_plan",
            "url",
            "desc",
            "purpose",
            "title",
            "zip_code",
            "addr_state",
            "earliest_cr_line",
            "initial_list_status",
            "last_pymnt_d",
            "next_pymnt_d",
            "last_credit_pull_d",
            "application_type",
            "verification_status_joint",
            "sec_app_earliest_cr_line",
            "hardship_flag",
            "hardship_type",
            "hardship_reason",
            "hardship_status",
            "hardship_start_date",
            "hardship_end_date",
            "payment_plan_start_date",
            "hardship_loan_status",
            "disbursement_method",
            "debt_settlement_flag",
            "debt_settlement_flag_date",
            "settlement_status",
            "settlement_date",
        ],
    },
    "rejected_2007_to_2018Q4.csv": {
        "float_cols": ["Amount Requested", "Risk_Score", "Policy Code"],
        "object_cols": [
            "Application Date",
            "Loan Title",
            "Debt-To-Income Ratio",
            "Zip Code",
            "State",
            "Employment Length",
        ],
    },
}


def load_csv_compressed(
        fp: typing.Union[str, Path],
        nrows: typing.Optional[int] = None,
        usecols: typing.Optional[typing.List[str]] = None,
):
    fp = Path(__file__).parent.joinpath(fp)
    col_dict = filename_to_cols[fp.name]

    dtypes = {col: np.dtype("float32") for col in col_dict["float_cols"]}
    for col in col_dict["object_cols"]:
        dtypes[col] = np.dtype("O")

    return pd.read_csv(fp, dtype=dtypes, nrows=nrows, usecols=usecols)


def capcurve(y_values, y_preds_proba, name='CatBoost', percent=.5, save:bool = True):
    num_pos_obs = int(np.sum(y_values))
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x': [0, rate_pos_obs, 1], 'y': [0, 1, 1]})
    xx = np.arange(num_count) / float(num_count - 1)

    y_cap = np.c_[y_values, y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(level=y_cap_df_s.index.names, drop=True)

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count - 1])  # add the first curve point (0,0) : for xx=0 we have yy=0

    row_index = int(np.trunc(num_count * percent))

    val_y1 = yy[row_index]
    val_y2 = yy[row_index + 1]
    if val_y1 == val_y2:
        val = val_y1 * 1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index + 1]
        val = val_y1 + ((val_x2 - percent) / (val_x2 - val_x1)) * (val_y2 - val_y1)

    sigma_ideal = 1 * xx[num_pos_obs - 1] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
    sigma_model = integrate.simps(yy, xx)
    sigma_random = integrate.simps(xx, xx)

    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ideal['x'], ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx, yy, color='red', label=name)
    ax.plot(xx, xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1,
            label=f'{val * 100:.2f} % of positive obs at ' + str(percent * 100) + '%')

    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
    plt.title("CAP Curve")
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()
    plt.savefig(f'{name}.svg')
