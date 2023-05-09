# SATD Tracker

SATD Tracker is a tool that extracts Self Admitted Technical Debts (SATDs) from a Git project directory and saves them into a CSV file. This tool was presented in our paper titled [Automated Self-Admitted Technical Debt Tracking at Commit-Level: A Language-independent Approach](https://arxiv.org/abs/2304.07829) published at the [TechDebt 2023 conference](https://2023.techdebtconf.org/details/TechDebt-2023-short-papers/2/Automated-Self-Admitted-Technical-Debt-Tracking-at-Commit-Level-A-Language-independe).

## Features

For each extracted SATD, the tool reports the following information:

- created_in_file: The file path which the SATD created in.
- last_appeared_in_file: The file path which the SATD appeared in for the last time before it was deleted or if we traced it to the last commit.
- created_in_line: The SATD line number when it was created.
- last_appeared_in_line: The SATD line number when it appeared for the last time.
- created_in_commit: The commit in which the SATD was created in.
- deleted_in_commit: The commit in which the SATD was deleted in.
- created_at_date: The date-time of the commit that the SATD was created in.
- deleted_at_date: The date-time of the commit that the SATD was deleted in.
- content: The whole line content in which we detect the SATD in.

In case a SATD line is updated in one or more commits, we also provide:

- deleted_in_lines
- created_in_lines
- updated_in_commits

## Update

We made an update to the tool after presenting it in the conference. To improve performance, we eliminated the usage of vcsSHARK tool and used our own code to explore the Git repository and extract the required information.

## Usage

1. Install the python packages listed in the requirements.txt:

```
pip install -r requirements.txt
```

2. Clone the repository to your local storage. For example:

```
git clone https://github.com/apache/commons-math.git
```

3. Run the following command to extract the SATDs from the locally saved repository:

```
python SATD_Tracker.py -path "/content/projects/commons-math" -output "commons-math-SATD.csv"
```

## Citation

If you use this tool in your research, please cite our paper:

```
@inproceedings{SATD-Tracker,
   title={Automated Self-Admitted Technical Debt Tracking at Commit-Level: A Language-independent Approach},
   author={Mohammad Sadegh Sheikhaei, Yuan Tian},
   booktitle={Proceedings of the TechDebt 2023 conference},
   year={2023}
}
```
