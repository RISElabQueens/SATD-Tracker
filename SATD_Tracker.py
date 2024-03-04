import sys
import time
import argparse
import pandas as pd
from git import Repo
from io import StringIO
from unidiff import PatchSet
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import shutil
import os

nltk.download('stopwords', quiet=True)
cachedStopWords = set(stopwords.words("english")) # we cache it for better performance (this way it runs much faster!)


def get_master_commits(repo):
    # Get the master branch
    master_branch = repo.head.reference

    # Get all master commits
    master_commits_raw = list(repo.iter_commits(master_branch))
    print("Total commits in master branch:", len(master_commits_raw))

    # sort the commits by date
    master_commits_raw = sorted(master_commits_raw, key=lambda x: x.committed_datetime)

    # remove master commits that are not the first parent
    master_commits = [master_commits_raw[-1]]
    while len(master_commits[-1].parents)>0:
        master_commits.append(master_commits[-1].parents[0])
    master_commits.reverse()
    print("Commits in the direct lineage of master branch (i.e., commits that are the first parent of next commit):", len(master_commits))
    return master_commits


def get_files_hunks_for_a_commit(repo , commit, file=None):
    EMPTY_TREE_SHA = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
    if len(commit.parents)==0:
        parent = EMPTY_TREE_SHA
    else:
        parent = commit.parents[0] # ??? is parents[0] always the parent in the master branch
    if file is None:
        uni_diff_text = repo.git.diff(parent, commit)
    else:
        uni_diff_text = repo.git.diff(parent, commit, file)
    patch_set = PatchSet(StringIO(uni_diff_text)) # we have to use PatchSet library because diff in gitpython provides all information in string (even with parent.diff(commit), the diff for each file/hunk is string)
    file_patchindex = {} # the index of files in patch_set
    for patched_file in patch_set:
        file_patchindex[patched_file.path] = len(file_patchindex)
    
    files_hunks = {}
    oldpath_newpath = {}    # we need the file name in the time (commit) that the SATD action is occured 
    if parent==EMPTY_TREE_SHA:
        if file is None:
            diff = commit.diff(EMPTY_TREE_SHA, create_patch=True) # we have to use this method because PatchSet doesn't provide the file renames correctly
        else:
            diff = commit.diff(EMPTY_TREE_SHA, file, create_patch=True) # we have to use this method because PatchSet doesn't provide the file renames correctly
    else:
        if file is None:
            diff = parent.diff(commit) # we have to use this method because PatchSet doesn't provide the file renames correctly
        else:
            diff = parent.diff(commit, file) # we have to use this method because PatchSet doesn't provide the file renames correctly

    for patch in diff:
        ab_path = patch.b_path
        if (patch.b_path != None) and (patch.b_path in file_patchindex):
            indx = file_patchindex[patch.b_path]
            hunks = patch_set[indx]
        elif patch.a_path in file_patchindex:
            ab_path = patch.a_path
            indx = file_patchindex[patch.a_path]
            hunks = patch_set[indx]
        else:
            print('Nither patch.b_path nor patch.a_path found in file_patchindex:')
            print('a_path:',patch.a_path)
            print('b_path:',patch.b_path)
            hunks = []

        files_hunks[ab_path] = hunks
        if patch.change_type in ['R']: # C: copy   R:rename  include C??????
            oldpath_newpath[patch.rename_from] = patch.b_path

    return oldpath_newpath, files_hunks


# I tried to change this method to run it faster by multi-threading, 
# but failed, likely due to Global Interpreter Lock (GIL) in Python
def get_files_hunks_for_all_commits(repo, commits, file=None):
    files_hunks = {}
    i=0
    for commit in commits:
        if i%500 == 0:
            print('Processed commits:',i, 'of', len(commits)) 
        oldpath_newpath, fh = get_files_hunks_for_a_commit(repo, commit, file=file)
        # update the keys in files_hunks according to oldpath_newpath
        for oldp,newp in oldpath_newpath.items():
            # if newp already exists in files_hunks (meaning that there was a file with the same name but deleted before this commit), we convert its key to filepath_#_deletionCommit
            if newp in files_hunks:
                # print('newp already exists in files_hunks ===>',newp)
                hexsha = files_hunks[newp][-1]['commit'].hexsha
                files_hunks[newp+'_#_'+hexsha] = files_hunks.pop(newp)
                # print('we change that key to:',newp+'_#_'+hexsha)
            if oldp in files_hunks:
                files_hunks[newp] = files_hunks.pop(oldp)
            # else:
                # print("oldpath not found in files_hunks:",oldp)
        # add the new fh to files_hunks
        for f,h in fh.items():
            if f not in files_hunks:
                files_hunks[f] = []
            files_hunks[f].append({'commit':commit, 'file':f, 'hunks':h})
        i += 1
    print('Processed commits:',i, 'of', len(commits)) 
    return files_hunks
    

def get_single_line_comment(line, file_extension):
    comment = ""
    file_extension = file_extension.lower()
    if file_extension in ['py', 'php', 'rb', 'sql', 'pl', 'r']:
        if "#" in line:
            comment = line.split("#", 1)[1]
            
    elif file_extension in ['java', 'js', 'c', 'cpp', 'h', 'cs', 'php', 'swift', 'go', 'kt', 'kts', 'scala', 'rs']:
        if "//" in line:
            comment = line.split("//", 1)[1].strip()
        # elif "/*" in line:
        #     comment = line.split("/*", 1)[1].split("*/", 1)[0].strip()
    elif file_extension == 'm':
        if "%" in line:
            comment = line.split("%", 1)[1]    
    return comment

# This code is based on MAT SATD detection introduced in paper "How Far HaveWe Progressed in Identifying Self-admitted Technical Debts? A Comprehensive Empirical Study"
# https://github.com/Naplues/MAT/blob/master/src/main/methods/Mat.java
# I updated it to extract the comment from any programming language, and then apply the MAT approach
def ismatch_MAT(string, file_extension):
    comment =  get_single_line_comment(string, file_extension)
    comment = comment.lower().replace("'","")
    tokens = comment.split(' ')
    for token in tokens:
        for keyword in ['todo','fixme','hack']:
            if token.startswith(keyword) or token.endswith(keyword):
                return True
            if token=='xxx':
                return True
    return False


# In this version we save more information about SATDs (i.e. hunk_index and SATD's context (its prev and next line))
# This information will be used to detect the following SATD of each SATD (if exists)
def get_raw_SATDs(hunks, steps, file_extension):
    newLineWarning = 'No newline at end of file' # this is a synthetic static warning that I should ignore, because it is not part of the user code
    # an example for newLineWarning:
        # file name: helix-core/src/test/java/org/apache/helix/alerts/TestEvaluateAlerts.java
        # https://github.com/apache/helix/commit/f414aad4c9b26fc767eaf373f7691f8e0487a598  --> see deleted line 395
    satds = []
    if steps<=0:
        steps = len(hunks)
    for i in range(steps): # iterate commits
        ####### detect deleted SATDs ########
        for j in range(len(hunks[i]['hunks'])): # iterate hunks for deleted SATDs
            hunk = hunks[i]['hunks'][j]
            lines = str(hunk).split('\n')[1:] # we ignore the first line because it is not part of the code (e.g. "@@ -18,12 +18,8 @@")
            l = 0
            for n in range(len(lines)):
                prevLine = lines[n-1] if n>0 else ''
                line = lines[n]
                nextLine = lines[n+1] if n<len(lines)-1 else ''
                if len(line)>0 and line[0]=='-':
                    #if 'todo' in line.lower():
                    if ismatch_MAT(line, file_extension):
                        codeLine = hunk.source_start + l
                        for satd in satds:
                            if satd['deleted_in_commit']==None and satd['line']==codeLine:
                                if n!=0 and len(prevLine)>0 and prevLine[0]=='-':
                                    satd['prev_line_content'] = prevLine[1:].strip()
                                if n!=len(lines)-1 and len(nextLine)>0 and nextLine[0]=='-':
                                    satd['next_line_content'] = nextLine[1:].strip()
                                satd['deleted_in_commit']=hunks[i]['commit']
                                satd['last_appeared_in_file']=hunks[i]['file'] # if the SATD is deleted from a file, and at the same time the file is renamed, we consider the new name (i.e., hunks[i]['file']), because in case of SATD update, we will find it in the new file name
                                satd['deleted_in_hunk']=j # hunk_index (in the old version that used vcsSHARK data, we used hunk_id, but now we use the hunk_index)
                if len(line)==0 or (len(line)>0 and line[0]!='+'):
                    l += 1
        ####### update SATD line numbers ########
        for satd in satds:
            satd['line_change'] = 0
        ## check hunks for deleted lines
        for j in range(len(hunks[i]['hunks'])):
            hunk = hunks[i]['hunks'][j]
            for satd in satds:
                if satd['deleted_in_commit']==None and satd['created_in_commit']!=hunks[i]['commit']: # probably no need the second condition but lets keep it just in case
                    if satd['line']>=hunk.source_start:
                        if satd['line']<hunk.source_start+hunk.source_length:
                            satd['line_change'] -= satd['line']-hunk.source_start
                        else:
                            satd['line_change'] -= hunk.source_length
        for satd in satds:
            satd['line_before_update'] = satd['line'] # it will be used to sort satds in the next step "check hunks for added lines"
            satd['line'] += satd['line_change']  # as the target_start is based on updated line numbers (after deletion) we have to first update line numbers here before checking for added lines
        ## check hunks for added lines
        for hunk in sorted(hunks[i]['hunks'], key=lambda item: item.target_start):
            lines = str(hunk).split('\n')[1:]
            l = 0
            unchanged_satds_l = []  # number of lines to be added to unchanged_satds (unchanged_satds are the satds that appear in the diff but their line doesn't start with - or +)
            for line in lines:
                #if len(line)>0 and line[0]!='-' and line[0]!='+' and 'todo' in line.lower():
                if len(line)>0 and line[0]!='-' and line[0]!='+' and ismatch_MAT(line, file_extension):
                    unchanged_satds_l.append(l)
                if len(line)==0 or (len(line)>0 and line[0]!='-' and line.strip()!=newLineWarning): 
                    l+=1
            for satd in sorted(satds, key=lambda item: item['line_before_update']):
                if satd['deleted_in_commit']==None and satd['created_in_commit']!=hunks[i]['commit']: # maybe no need the second condition but lets keep it just in case
                    if satd['line']>=hunk.target_start:
                        if satd['line']<hunk.target_start+hunk.target_length and len(unchanged_satds_l)>0:  # ?? it was hunk['new_start']+hunk['new_start'] but I think that was my mistake,
                                                                                                            # so I updated it to hunk['new_start']+hunk['new_lines'], although the results remained unchange!
                            satd['line'] += unchanged_satds_l[0]
                            unchanged_satds_l = unchanged_satds_l[1:]
                        else:
                            satd['line'] += hunk.target_length
        ####### find new SATDs ########
        for j in range(len(hunks[i]['hunks'])): # iterate hunks for created SATDs
            hunk = hunks[i]['hunks'][j]
            lines = str(hunk).split('\n')[1:]
            l = 0
            for n in range(len(lines)):
                prevLine = lines[n-1] if n>0 else ''
                line = lines[n]
                nextLine = lines[n+1] if n<len(lines)-1 else ''
                if len(line)>0 and line[0]=='+':
                    #if 'todo' in re.findall(r"[\w']+|[.,!?;]", line.lower()):
                    if ismatch_MAT(line, file_extension):
                        codeLine = hunk.target_start + l
                        satd = {'created_in_file':hunks[i]['file'], 'last_appeared_in_file':hunks[i]['file'], 'created_in_line':codeLine, 'line':codeLine, 'created_in_commit':hunks[i]['commit'], 'deleted_in_commit':None, 'created_in_hunk':j, 'deleted_in_hunk':None, 'content':line[1:].strip()}
                        if n!=0 and len(prevLine)>0 and prevLine[0]=='+':
                            satd['prev_line_content'] = prevLine[1:].strip()
                        else:
                            satd['prev_line_content'] = ''
                        if n!=len(lines)-1 and len(nextLine)>0 and nextLine[0]=='+':
                            satd['next_line_content'] = nextLine[1:].strip()    
                        else:
                            satd['next_line_content'] = ''
                        satds.append(satd)
                if len(line)==0 or (len(line)>0 and line[0]!='-' and line.strip()!=newLineWarning):
                    l += 1
    for satd in satds:
        if 'line_change' in satd:
            del satd['line_change']
        if 'line_before_update' in satd:
            del satd['line_before_update']
    return satds


# if target_commit=None: Extract all satds in the master branch of a project
# if target_commit!=None: Extract all satds in a sequence of commits (preferably in the master branch) of a project that leads to a target commit
def get_project_SATDs(files_hunks, file_extensions, target_commit=None):
    num_files = len(files_hunks)
    print('number of files in project:',num_files)

    filecs_satds = {}
    start_time = time.time()
    
    i = 0
    for filec, sortedActions in files_hunks.items():  # filec is file or file_commit
        i += 1
        file = filec.split('_#_')[0]
        if i%500==0:
            #print('number of processed files: %d  file(_commit)=%s file_name=%s                         \r'%(i,filec, files_name[file_id]), end="")
            print('number of processed files:', i)
        if file.split('.')[-1] in file_extensions: 
            filecs_satds[filec] = get_raw_SATDs(files_hunks[filec], 0, file.split('.')[-1]) 
    print('number of processed files:', i)
    return filecs_satds

def SATDs_to_dataframe(filecs_satds):
    last_appeared_in_file = []
    #file_deleted_in_commit = []
    created_in_file = []
    created_in_line = []
    line = []
    created_in_commit = []
    deleted_in_commit = []
    created_at_date = []
    deleted_at_date = []
    created_in_hunk = []
    deleted_in_hunk = []
    content = []
    prev_line_content = []
    next_line_content = []
    for filec,satds in filecs_satds.items():
        for satd in satds:
            last_appeared_in_file.append(satd['last_appeared_in_file'])
            #last_appeared_in_file.append(filec.split('_#_')[0])  # TODO: change _#_ to FILE_COMMIT_SEP
            #file_deleted_in_commit.append(filec.split('_#_')[1] if '_#_' in filec else None)
            created_in_file.append(satd['created_in_file'])
            created_in_line.append(satd['created_in_line'])
            line.append(satd['line'])
            created_in_commit.append(satd['created_in_commit'].hexsha)
            deleted_in_commit.append('' if satd['deleted_in_commit'] is None else satd['deleted_in_commit'].hexsha)
            created_at_date.append(satd['created_in_commit'].committed_datetime)
            deleted_at_date.append('' if satd['deleted_in_commit'] is None else satd['deleted_in_commit'].committed_datetime)
            created_in_hunk.append(satd['created_in_hunk'])
            deleted_in_hunk.append('' if satd['deleted_in_hunk'] is None else satd['deleted_in_hunk'])
            content.append(satd['content'])
            prev_line_content.append(satd['prev_line_content'])
            next_line_content.append(satd['next_line_content'])
    df = pd.DataFrame(
    {'created_in_file': created_in_file,
     'last_appeared_in_file': last_appeared_in_file,
     #'lastfile_deleted_in_commit': file_deleted_in_commit,
     'created_in_line': created_in_line,
     'last_appeared_in_line': line,
     'created_in_commit': created_in_commit,
     'deleted_in_commit': deleted_in_commit,
     'created_at_date': pd.to_datetime(created_at_date, utc=True),
     'deleted_at_date': pd.to_datetime(deleted_at_date, utc=True),
     'created_in_hunk': created_in_hunk,
     'deleted_in_hunk': deleted_in_hunk,
     'content': content,
     'prev_line_content': prev_line_content,
     'next_line_content': next_line_content
    })
    return df    
    

def add_followingSatdCandidates(df):
    created_in_file = df['created_in_file']
    last_appeared_in_file = df['last_appeared_in_file']
    created_in_commit = df['created_in_commit']
    deleted_in_commit = df['deleted_in_commit']
    followingSatdCandidates = len(last_appeared_in_file) * ['']
    for i in range(len(last_appeared_in_file)):
        for j in range(len(last_appeared_in_file)):
            if i!=j and last_appeared_in_file[i]==created_in_file[j] and deleted_in_commit[i]==created_in_commit[j]:
                if followingSatdCandidates[i]=='':
                    followingSatdCandidates[i] = str(j)
                else:
                    followingSatdCandidates[i] += ', '+str(j)
    df['followingSatdCandidates'] = followingSatdCandidates
    return df

def my_tokenizer(text, remove_singleCharWords=False, remove_stopwords=False, stemming=False):
    #tokens = text.split(' ')
    tokens = re.findall(r"@\w+|\w+|\S", text)
    #tokens = re.findall(r"[A-Za-z]+|[0-9]+|\S", text)
    #tokens = [word for word in nltk.word_tokenize(text)]
    
    if remove_singleCharWords:
        tokens = [word for word in tokens if len(word)>1]
    if remove_stopwords:
        tokens = [word for word in tokens if not word.lower() in cachedStopWords and len(word)>1]
    if stemming:
        tokens = [stemmer.stem(item) for item in tokens]
    return tokens

# get two text and return the jaccard similarity
def get_jaccard_sim(text1, text2, remove_singleCharWords, remove_stopwords):
    if type(text1)==str:
        text1_words = set(my_tokenizer(text1, remove_singleCharWords=remove_singleCharWords, remove_stopwords=remove_stopwords))
    elif type(text1)==dict:
        text1_words = set(text1.keys())
    if type(text2)==str:
        text2_words = set(my_tokenizer(text2, remove_singleCharWords=remove_singleCharWords, remove_stopwords=remove_stopwords))
    elif type(text2)==dict:
        text2_words = set(text2.keys())
    if remove_stopwords:
        text1_words = text1_words.difference(cachedStopWords)  # remove stopwords
        text2_words = text2_words.difference(cachedStopWords)  # remove stopwords
    intersec_words = text1_words.intersection(text2_words)
    if len(text1_words)==0 and len(text2_words)==0:
        return 0
    return float(len(intersec_words)) / (len(text1_words) + len(text2_words) - len(intersec_words))

def get_jaccard_sim_matrix(strList1, strList2, remove_singleCharWords, remove_stopwords):
    matrix = np.zeros((len(strList1), len(strList2)))
    for i in range(len(strList1)):
        for j in range(len(strList2)):
            matrix[i,j] = get_jaccard_sim(strList1[i], strList2[j], remove_singleCharWords, remove_stopwords)
    return matrix

def get_hunk_sim_matrix(hunkList1, hunkList2):
    matrix = np.zeros((len(hunkList1), len(hunkList2)))
    for i in range(len(hunkList1)):
        for j in range(len(hunkList2)):
            matrix[i,j] = 1 if hunkList1[i]==hunkList2[j] else 0
    return matrix

# for a specific file and a specific commit, it gets the information (e.g. prevLine, nextLine, str, hunk#) of deleted and inserted SATDs as List1 and List2 respectively
# it returns an array of n by 2 that n is the number of found matches. For each match, it stores the index of deleted SATD in list1 and the index of new SATD in list2.
def string_match(file_commit_id, strList1, strList2, prevList1, prevList2, nextList1, nextList2, hunkList1, hunkList2, strWeight, prevWeight, nextWeight, hunkWeight, threshold, remove_singleCharWords, remove_stopwords):
    if not hasattr(string_match, "cache"):
        string_match.cache = {}
    if file_commit_id in string_match.cache:
        matrixStr = string_match.cache[file_commit_id]['matrixStr']
        matrixPrev = string_match.cache[file_commit_id]['matrixPrev']
        matrixNext = string_match.cache[file_commit_id]['matrixNext']
        matrixHunk = string_match.cache[file_commit_id]['matrixHunk']
    else:
        matrixStr = get_jaccard_sim_matrix(strList1, strList2, remove_singleCharWords, remove_stopwords)
        matrixPrev = get_jaccard_sim_matrix(prevList1, prevList2, remove_singleCharWords, remove_stopwords)
        matrixNext = get_jaccard_sim_matrix(nextList1, nextList2, remove_singleCharWords, remove_stopwords)
        matrixHunk = get_hunk_sim_matrix(hunkList1, hunkList2)
        string_match.cache[file_commit_id] = {}
        string_match.cache[file_commit_id]['matrixStr'] = matrixStr
        string_match.cache[file_commit_id]['matrixPrev'] = matrixPrev
        string_match.cache[file_commit_id]['matrixNext'] = matrixNext
        string_match.cache[file_commit_id]['matrixHunk'] = matrixHunk

    matrix = strWeight*matrixStr + prevWeight*matrixPrev + nextWeight*matrixNext + hunkWeight*matrixHunk
    matches=[]
    while np.amax(matrix)>threshold:
        maxInd = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        matches.append(maxInd)
        matrix[maxInd[0],:] = 0
        matrix[:,maxInd[1]] = 0
    return matches

# returns a dictionary of files - commits - matchingSatds
# matchingSatds contains the information of deleted and inserted Satds (like the Satd line, and its prevous and next lines) in that file-commit
def get_files_commits_matchingSatds(df):
    files_commits_matchingSatds = {}
    for index, row in df.iterrows():
        if len(row['followingSatdCandidates'])>0:
            if row['last_appeared_in_file'] not in files_commits_matchingSatds:
                files_commits_matchingSatds[row['last_appeared_in_file']] = {}
            for candid in row['followingSatdCandidates'].split(','):
                if row['deleted_in_commit'] not in files_commits_matchingSatds[row['last_appeared_in_file']]:
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']] = {'indList1':[], 'indList2':[], 'strList1':[], 'strList2':[], 'prevList1':[], 'prevList2':[], 'nextList1':[], 'nextList2':[], 'hunkList1':[], 'hunkList2':[]}
                candid = int(candid.strip())
                if index not in files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['indList1']:
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['indList1'].append(index)
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['strList1'].append(row['content'])
                    #files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['strList1'].append(re.split(';\s*/*', row['content'])[-1]) # takes only the SATD part
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['prevList1'].append(row['prev_line_content'])
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['nextList1'].append(row['next_line_content'])
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['hunkList1'].append(row['deleted_in_hunk'])
                if candid not in files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['indList2']:
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['indList2'].append(candid)
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['strList2'].append(df.iloc[candid, df.columns.get_loc('content')])
                    #files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['strList2'].append(re.split(';\s*/*',df.iloc[candid, df.columns.get_loc('content')])[-1]) # takes only the SATD part
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['prevList2'].append(df.iloc[candid, df.columns.get_loc('prev_line_content')])
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['nextList2'].append(df.iloc[candid, df.columns.get_loc('next_line_content')])
                    files_commits_matchingSatds[row['last_appeared_in_file']][row['deleted_in_commit']]['hunkList2'].append(df.iloc[candid, df.columns.get_loc('created_in_hunk')])
    return files_commits_matchingSatds 
 
def add_followingSatdByGreedy(df, files_commits_matchingSatds, strWeight, prevWeight, nextWeight, hunkWeight, threshold):
    df['followingSatdByGreedy'] = ''
    for file, commits_matchingSatds in files_commits_matchingSatds.items():
        for commit, matchingSatds in commits_matchingSatds.items():
            file_commit_id = file+'_'+commit
            matches = string_match(file_commit_id, matchingSatds['strList1'], matchingSatds['strList2'], matchingSatds['prevList1'], matchingSatds['prevList2'], matchingSatds['nextList1'], matchingSatds['nextList2'], matchingSatds['hunkList1'], matchingSatds['hunkList2'], strWeight, prevWeight, nextWeight, hunkWeight, threshold, False, False)
            for ind in matchingSatds['indList1']:
                df.iloc[ind, df.columns.get_loc('followingSatdByGreedy')] = '-'
            for match in matches:
                df.iloc[matchingSatds['indList1'][match[0]], df.columns.get_loc('followingSatdByGreedy')] = matchingSatds['indList2'][match[1]]
    return df


# After finding the following SATDs, we can merge them and delete false positive rows.
# When we merge them, we need to update some columns like deleted_in_commit, line, and content
def merge_followingSATDs_in_dataframe(df, followingSatdColumn, deleteFollowingSATDs):
    df['followed_by'] = '' # the list of following SATDs. For example, if 11 follwos 10, and 12 follwos 11, this field will be "11,12" for row 10
    df['deleted_in_lines'] = '' # it has the list of middle deleted lines. i.e. the lines when the SATD were updated.
    df['created_in_lines'] = '' # it has the list of middle created lines. i.e. the lines when the SATD were updated.
    df['updated_in_commits'] = '' # it has the list of middle commits. 
    deleteIndices = []
    for index, row in df.iterrows():
        deleted_in_lines=[]
        created_in_lines=[]
        content=row['content']
        updated_in_commits = []
        findex = index
        while df.iloc[findex, df.columns.get_loc(followingSatdColumn)] not in ['','-']:
            deleted_in_lines.append(df.iloc[findex, df.columns.get_loc('last_appeared_in_line')])
            findex = df.iloc[findex, df.columns.get_loc(followingSatdColumn)]
            df.iloc[index, df.columns.get_loc('followed_by')] += str(findex) + ','
            created_in_lines.append(df.iloc[findex, df.columns.get_loc('created_in_line')])
            content += '\n' + df.iloc[findex, df.columns.get_loc('content')]
            updated_in_commits.append(df.iloc[findex, df.columns.get_loc('created_in_commit')])
            deleteIndices.append(findex)
        if findex!=index:
            df.iloc[index, df.columns.get_loc('deleted_in_commit')] = df.iloc[findex, df.columns.get_loc('deleted_in_commit')]
            df.iloc[index, df.columns.get_loc('deleted_in_hunk')] = df.iloc[findex, df.columns.get_loc('deleted_in_hunk')]
            df.iloc[index, df.columns.get_loc('deleted_at_date')] = df.iloc[findex, df.columns.get_loc('deleted_at_date')]
            df.iloc[index, df.columns.get_loc('last_appeared_in_line')] = df.iloc[findex, df.columns.get_loc('last_appeared_in_line')]
            df.iloc[index, df.columns.get_loc('deleted_in_lines')] = str(deleted_in_lines)
            df.iloc[index, df.columns.get_loc('created_in_lines')] = str(created_in_lines)
            df.iloc[index, df.columns.get_loc('content')] = content
            df.iloc[index, df.columns.get_loc('updated_in_commits')] = str(updated_in_commits)
            if 'deletedInMaster' in df.columns:
                df.iloc[index, df.columns.get_loc('deletedInMaster')] = df.iloc[findex, df.columns.get_loc('deletedInMaster')]

    if deleteFollowingSATDs:
        df = df.drop(index=deleteIndices)
        df = df.reset_index(drop=True)
        df = df.drop(columns=['followingSatdCandidates', 'followingSatdByGreedy', 'followingSatdCandidatesByFileRename', 'followingSatdByFileRename', 'followingSatdByHeuristics', 'followed_by'], errors='ignore')
    df = df.drop(columns=['prev_line_content', 'next_line_content'])
    return df    


def parse_arguments():
    def ensure_trailing_slash(path):
        """Ensure the path ends with a slash."""
        return path if path.endswith('/') else path + '/'

    parser = argparse.ArgumentParser(description='SATD-Tracker parameters')
    parser.add_argument('--repo', type=str, default="https://github.com/apache/commons-math.git",
                        help='Repository path (local directory or GitHub URL) [default: https://github.com/apache/commons-math.git]')
    parser.add_argument('--file-extensions', type=lambda s: [ext.strip() for ext in s.split(',')], default='py, php, rb, sql, pl, r, java, js, c, cpp, h, cs, swift, go, kt, kts, scala, rs, m',
                        help='File extensions, for example: py,java,cpp. [default: py, php, rb, sql, pl, r, java, js, c, cpp, h, cs, swift, go, kt, kts, scala, rs, m]')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file [default: username___reponame_SATD.csv or directoryname_SATD.csv]')
    parser.add_argument('--output-path', type=ensure_trailing_slash, default='SATD/',
                        help='Output path [default: SATD/]')
    parser.add_argument('--provide-raw-SATD', action='store_true', default=False,
                        help='In addition to the final list of SATD, also provide raw SATD [default: False]')
    parser.add_argument('--downloading-path', type=ensure_trailing_slash, default='repositories/',
                        help='Downloading path for GitHub repositories [default: repositories/]')
    parser.add_argument('--delete-downloaded-repo', action='store_true', default=False,
                        help='Delete downloaded repository after extracting SATDs [default: False]')

    args = parser.parse_args()
    args.repo = args.repo.rstrip('/')
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.downloading_path, exist_ok=True)
    if args.output_file is None:
        repo_parts = args.repo.split('/')
        if args.repo.startswith('https://github.com'):
            username, reponame = repo_parts[-2], repo_parts[-1].replace('.git', '')
            args.output_file = f'{username}___{reponame}_SATD.csv'
            args.raw_output_file = f'{username}___{reponame}_rawSATD.csv'
        else:
            directoryname = repo_parts[-1]
            args.output_file = f'{directoryname}_SATD.csv'
            args.raw_output_file = f'{directoryname}_rawSATD.csv'
    else:
        if args.output_file.endswith('.csv') == False:
            args.output_file += '.csv'
        args.raw_output_file = args.output_file.replace('.csv','_raw.csv')
    return args

if __name__ == "__main__":
    args = parse_arguments()
    # Print all parameters
    print("\nConfiguration Parameters:")
    for arg, value in vars(args).items():
            print(f"{arg}: {value}")
    print()
    
    # download repository
    if args.repo.startswith('https://github.com'):
        print('Downloading repository...')
        repo_dir = args.downloading_path + args.repo.split('/')[-1].replace('.git', '')
        Repo.clone_from(args.repo, repo_dir)
        repo = Repo(repo_dir)
        print('Done')
    else:
        repo = Repo(args.repo)

    # get the sequence of commits in the master branch
    master_commits = get_master_commits(repo)
    
    # extract the sequence of hunks for all files
    print("\nExtract the sequence of hunks for all files...")
    files_hunks = get_files_hunks_for_all_commits(repo, master_commits)
    if None in files_hunks:
        del files_hunks[None]
        
    # Extract and track raw SATDs from files_hunks
    print("Extract and track raw SATDs from files_hunks...")
    files_satds = get_project_SATDs(files_hunks, args.file_extensions)
    df = SATDs_to_dataframe(files_satds)
    print("Number of extracted raw SATDs:", len(df))
    
    if 'followingSatd' in df.columns:
        df.followingSatd = df.followingSatd.fillna('')
    if 'prev_line_content' in df.columns:
        df.prev_line_content = df.prev_line_content.fillna('')
    if 'next_line_content' in df.columns:
        df.next_line_content = df.next_line_content.fillna('')
    
    # Find the following SATDs
    df = add_followingSatdCandidates(df)
    files_commits_matchingSatds = get_files_commits_matchingSatds(df)
    df = add_followingSatdByGreedy(df, files_commits_matchingSatds, 0.6, 0.2, 0, 0.2, 0.4) # these are the optimum weights we obtained through a grid search on a labeled dataset
    
    # save raw SATDs
    if args.provide_raw_SATD:
        df.to_csv(args.output_path + args.raw_output_file)
        print("The extracted raw SATDs saved in", args.output_path + args.raw_output_file)

    # merge the following SATDs
    df2 = df.copy()
    df2 = merge_followingSATDs_in_dataframe(df2, 'followingSatdByGreedy', True)
    df2 = df2.drop(columns=['created_in_hunk', 'deleted_in_hunk']) # no need to drop 'lastfile_deleted_in_commit'. We don't have it anymore.
    print("Number of deleted SATDs after merging them:",len(df)-len(df2))
    print("Final number of SATDs:",len(df2))
    df2.to_csv(args.output_path + args.output_file, encoding='utf-8', errors='replace')
    print("The extracted SATDs saved in", args.output_path + args.output_file)

    # delete the downloaded repository
    if args.repo.startswith('https://github.com') and args.delete_downloaded_repo:
        shutil.rmtree(repo_dir)