# SATD Tracker

SATD Tracker is a tool that extracts Self Admitted Technical Debts (SATDs) from a Git project directory and saves them into a CSV file. This tool was presented in our paper titled "Automated Self-Admitted Technical Debt Tracking at Commit-Level: A Language-independent Approach" published at the TechDebt 2023 conference.

## Features

For each extracted SATD, the tool reports the following information:

- createdInFile: The file path which the SATD created in.
- lastAppearedInFile: The file path which the SATD appeared in for the last time before it was deleted or if we traced it to the last commit.
- lastFileDeleteInCommit: If the last file which the SATD appeared in is deleted, it shows the corresponding commit.
- createdInLine: The SATD line number when it was created.
- lastAppearedInLine: The SATD line number when it appeared for the last time.
- createdInCommit: The commit in which the SATD was created in.
- deletedInCommit: The commit in which the SATD was deleted in.
- createdInDate: The date-time of the commit that the SATD was created in.
- deletedInDate: The date-time of the commit that the SATD was deleted in.
- createdInHunk: The hunk number in which the SATD was created in.
- deletedInHunk: The hunk number in which the SATD was deleted in.
- content: The whole line content in which we detect the SATD in.

In case a SATD line is updated in one or more commits, we also provide:

- deletedInLines
- createdInLines
- updatedInCommits

## Update

We made an update to the tool after presenting it in the conference. To improve performance, we eliminated the usage of vcsSHARK tool and used our own code to explore the Git repository and extract the required information.

## Usage

1. Clone the repository to your local storage. For example:

```
git clone https://github.com/apache/commons-math.git
```

2. Run the following command to extract the SATDs from the locally saved repository:

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
