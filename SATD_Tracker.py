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

nltk.download('stopwords')
cachedStopWords = set(stopwords.words("english")) # we cache it for better performance (this way it runs much faster!)


def get_master_commits(repo):
    # Get the master branch
    master_branch = repo.head.reference

    # Get all master commits
    master_commits_raw = list(repo.iter_commits(master_branch))

    # sort the commits by date
    master_commits_raw = sorted(master_commits_raw, key=lambda x: x.committed_datetime)

    # remove master commits that are not the first parent
    master_commits = [master_commits_raw[-1]]
    while len(master_commits[-1].parents)>0:
        master_commits.append(master_commits[-1].parents[0])
    master_commits.reverse()
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
        if patch.b_path in file_patchindex:
            indx = file_patchindex[patch.b_path]
            hunks = patch_set[indx]
        elif patch.a_path in file_patchindex:
            indx = file_patchindex[patch.a_path]
            hunks = patch_set[indx]
        else:
            print('Nither patch.b_path nor patch.a_path found in file_patchindex:')
            print('a_path:',patch.a_path)
            print('b_path:',patch.b_path)
            hunks = []

        files_hunks[patch.b_path] = hunks
        if patch.change_type in ['R']: # C: copy   R:rename  include C??????
            oldpath_newpath[patch.rename_from] = patch.b_path

    return oldpath_newpath, files_hunks


def get_files_hunks_for_all_commits(repo, commits, file=None):
    files_hunks = {}
    i=0
    for commit in commits:
        if i%500 == 0:
            print('Processed commits:',i) 
        oldpath_newpath, fh = get_files_hunks_for_a_commit(repo, commit, file=file)
        # update the keys in files_hunks according to oldpath_newpath
        for oldp,newp in oldpath_newpath.items():
            # if newp already exists in files_hunks (meaning that there was a file with the same name but deleted before this commit), we convert its key to filepath_#_deletionCommit
            if newp in files_hunks:
                #print('newp already exists in files_hunks ===>',newp)
                hexsha = files_hunks[newp][-1]['commit'].hexsha
                files_hunks[newp+'_#_'+hexsha] = files_hunks.pop(newp)
                #print('we change that key to:',newp+'_#_'+hexsha)
            if oldp in files_hunks:
                files_hunks[newp] = files_hunks.pop(oldp)
            #else:
                #print("oldpath not found in files_hunks:",oldp)
        # add the new fh to files_hunks
        for f,h in fh.items():
            if f not in files_hunks:
                files_hunks[f] = []
            files_hunks[f].append({'commit':commit, 'file':f, 'hunks':h})
        i += 1
    return files_hunks


# This code is based on MAT SATD detection introduced in paper "How Far HaveWe Progressed in Identifying Self-admitted Technical Debts? A Comprehensive Empirical Study"
# https://github.com/Naplues/MAT/blob/master/src/main/methods/Mat.java
def ismatch_MAT(string):
    strings = string.split('//',1)
    if len(strings)==2:
        comment = strings[1]
    else:
        return False
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
def get_raw_SATDs(hunks, steps):
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
                    if ismatch_MAT(line):
                        codeLine = hunk.source_start + l
                        for satd in satds:
                            if satd['deleted_in_commit']==None and satd['line']==codeLine:
                                if n!=0 and len(prevLine)>0 and prevLine[0]=='-':
                                    satd['prev_line_content'] = prevLine[1:].strip()
                                if n!=len(lines)-1 and len(nextLine)>0 and nextLine[0]=='-':
                                    satd['next_line_content'] = nextLine[1:].strip()
                                satd['deleted_in_commit']=hunks[i]['commit']
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
                if len(line)>0 and line[0]!='-' and line[0]!='+' and ismatch_MAT(line):
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
                    if ismatch_MAT(line):
                        codeLine = hunk.target_start + l
                        satd = {'created_in_file':hunks[i]['file'], 'created_in_line':codeLine, 'line':codeLine, 'created_in_commit':hunks[i]['commit'], 'deleted_in_commit':None, 'created_in_hunk':j, 'deleted_in_hunk':None, 'content':line[1:].strip()}
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
def get_project_SATDs(files_hunks, target_commit=None):
    num_files = len(files_hunks)
    print('number of files in project:',num_files)

    filecs_satds = {}
    start_time = time.time()
    
    i = 0
    for filec, sortedActions in files_hunks.items():  # filec is file or file_commit
        i += 1
        file = filec.split('_#_')[0]
        if i%500==0:
            print('number of processed files:', i)
        if file.endswith('.java'):  # ???? only java ????
            filecs_satds[filec] = get_raw_SATDs(files_hunks[filec], 0) 

    return filecs_satds

def SATDs_to_dataframe(filecs_satds):
    lastAppearedInFile = []
    fileDeleteInCommit = []
    createdInFile = []
    createdInLine = []
    line = []
    createdInCommit = []
    deletedInCommit = []
    createdInDate = []
    deletedInDate = []
    createdInHunk = []
    deletedInHunk = []
    content = []
    prevLineContent = []
    nextLineContent = []
    for filec,satds in filecs_satds.items():
        for satd in satds:
            lastAppearedInFile.append(filec.split('_#_')[0])  # TODO: change _#_ to FILE_COMMIT_SEP
            fileDeleteInCommit.append(filec.split('_#_')[1] if '_#_' in filec else None)
            createdInFile.append(satd['created_in_file'])
            createdInLine.append(satd['created_in_line'])
            line.append(satd['line'])
            createdInCommit.append(satd['created_in_commit'].hexsha)
            deletedInCommit.append('' if satd['deleted_in_commit'] is None else satd['deleted_in_commit'].hexsha)
            createdInDate.append(satd['created_in_commit'].committed_datetime)
            deletedInDate.append('' if satd['deleted_in_commit'] is None else satd['deleted_in_commit'].committed_datetime)
            createdInHunk.append(satd['created_in_hunk'])
            deletedInHunk.append('' if satd['deleted_in_hunk'] is None else satd['deleted_in_hunk'])
            content.append(satd['content'])
            prevLineContent.append(satd['prev_line_content'])
            nextLineContent.append(satd['next_line_content'])
    df = pd.DataFrame(
    {'createdInFile': createdInFile,
     'lastAppearedInFile': lastAppearedInFile,
     'lastFileDeleteInCommit': fileDeleteInCommit,
     'createdInLine': createdInLine,
     'lastAppearedInLine': line,
     'createdInCommit': createdInCommit,
     'deletedInCommit': deletedInCommit,
     'createdInDate': createdInDate,
     'deletedInDate': deletedInDate,
     'createdInHunk': createdInHunk,
     'deletedInHunk': deletedInHunk,
     'content': content,
     'prevLineContent': prevLineContent,
     'nextLineContent': nextLineContent
    })
    return df

def add_followingSatdCandidates(df):
    lastAppearedInFile = df['lastAppearedInFile']
    createdInCommit = df['createdInCommit']
    deletedInCommit = df['deletedInCommit']
    followingSatdCandidates = len(lastAppearedInFile) * ['']
    for i in range(len(lastAppearedInFile)):
        for j in range(len(lastAppearedInFile)):
            if i!=j and lastAppearedInFile[i]==lastAppearedInFile[j] and deletedInCommit[i]==createdInCommit[j]:
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

# for a specific file and a specific commit, it gets the iformation (e.g. prevLine, nextLine, str, hunk#) of deleted and inserted SATDs as List1 and List2 respectively
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
            if row['lastAppearedInFile'] not in files_commits_matchingSatds:
                files_commits_matchingSatds[row['lastAppearedInFile']] = {}
            for candid in row['followingSatdCandidates'].split(','):
                if row['deletedInCommit'] not in files_commits_matchingSatds[row['lastAppearedInFile']]:
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']] = {'indList1':[], 'indList2':[], 'strList1':[], 'strList2':[], 'prevList1':[], 'prevList2':[], 'nextList1':[], 'nextList2':[], 'hunkList1':[], 'hunkList2':[]}
                candid = int(candid.strip())
                if index not in files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['indList1']:
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['indList1'].append(index)
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['strList1'].append(row['content'])
                    #files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['strList1'].append(re.split(';\s*/*', row['content'])[-1]) # takes only the SATD part
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['prevList1'].append(row['prevLineContent'])
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['nextList1'].append(row['nextLineContent'])
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['hunkList1'].append(row['deletedInHunk'])
                if candid not in files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['indList2']:
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['indList2'].append(candid)
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['strList2'].append(df.iloc[candid, df.columns.get_loc('content')])
                    #files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['strList2'].append(re.split(';\s*/*',df.iloc[candid, df.columns.get_loc('content')])[-1]) # takes only the SATD part
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['prevList2'].append(df.iloc[candid, df.columns.get_loc('prevLineContent')])
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['nextList2'].append(df.iloc[candid, df.columns.get_loc('nextLineContent')])
                    files_commits_matchingSatds[row['lastAppearedInFile']][row['deletedInCommit']]['hunkList2'].append(df.iloc[candid, df.columns.get_loc('createdInHunk')])
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
# When we merge them, we need to update some columns like deletedInCommit, line, and content
def merge_followingSATDs_in_dataframe(df, followingSatdColumn, deleteFollowingSATDs):
    df['followedBy'] = '' # the list of following SATDs. For example, if 11 follwos 10, and 12 follwos 11, this field will be "11,12" for row 10
    df['deletedInLines'] = '' # it has the list of middle deleted lines. i.e. the lines when the SATD were updated.
    df['createdInLines'] = '' # it has the list of middle created lines. i.e. the lines when the SATD were updated.
    df['updatedInCommits'] = '' # it has the list of middle commits. 
    deleteIndices = []
    for index, row in df.iterrows():
        deletedInLines=[]
        createdInLines=[]
        content=row['content']
        updatedInCommits = []
        findex = index
        while df.iloc[findex, df.columns.get_loc(followingSatdColumn)] not in ['','-']:
            deletedInLines.append(df.iloc[findex, df.columns.get_loc('lastAppearedInLine')])
            findex = df.iloc[findex, df.columns.get_loc(followingSatdColumn)]
            df.iloc[index, df.columns.get_loc('followedBy')] += str(findex) + ','
            createdInLines.append(df.iloc[findex, df.columns.get_loc('createdInLine')])
            content += '\n' + df.iloc[findex, df.columns.get_loc('content')]
            updatedInCommits.append(df.iloc[findex, df.columns.get_loc('createdInCommit')])
            deleteIndices.append(findex)
        if findex!=index:
            df.iloc[index, df.columns.get_loc('deletedInCommit')] = df.iloc[findex, df.columns.get_loc('deletedInCommit')]
            df.iloc[index, df.columns.get_loc('deletedInHunk')] = df.iloc[findex, df.columns.get_loc('deletedInHunk')]
            df.iloc[index, df.columns.get_loc('deletedInDate')] = df.iloc[findex, df.columns.get_loc('deletedInDate')]
            df.iloc[index, df.columns.get_loc('lastAppearedInLine')] = df.iloc[findex, df.columns.get_loc('lastAppearedInLine')]
            df.iloc[index, df.columns.get_loc('deletedInLines')] = str(deletedInLines)
            df.iloc[index, df.columns.get_loc('createdInLines')] = str(createdInLines)
            df.iloc[index, df.columns.get_loc('content')] = content
            df.iloc[index, df.columns.get_loc('updatedInCommits')] = str(updatedInCommits)
            if 'deletedInMaster' in df.columns:
                df.iloc[index, df.columns.get_loc('deletedInMaster')] = df.iloc[findex, df.columns.get_loc('deletedInMaster')]

    if deleteFollowingSATDs:
        df = df.drop(index=deleteIndices)
        df = df.reset_index(drop=True)
        df = df.drop(columns=['followingSatdCandidates', 'followingSatdByGreedy', 'followingSatdCandidatesByFileRename', 'followingSatdByFileRename', 'followingSatdByHeuristics', 'followedBy'], errors='ignore')
    df = df.drop(columns=['prevLineContent', 'nextLineContent'])
    return df
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="path to the git project folder")
    parser.add_argument("-output", default="SATD-merged.csv", help="path to output csv file")
    args = parser.parse_args()

    if not args.path:
        print("Error: Git project path not provided.")
        sys.exit(1)

    # get the repository
    repo = Repo(args.path)

    # get the sequence of commits in the master branch
    master_commits = get_master_commits(repo)
    
    # extract the sequence of hunks for all files
    print("Extract the sequence of hunks for all files...")
    files_hunks = get_files_hunks_for_all_commits(repo, master_commits)
    if None in files_hunks:
        del files_hunks[None]
        
    # Extract and track raw SATDs from files_hunks
    print("Extract and track raw SATDs from files_hunks...")
    files_satds = get_project_SATDs(files_hunks)
    df = SATDs_to_dataframe(files_satds)
    print("Number of extracted raw SATDs:", len(df))
    
    if 'followingSatd' in df.columns:
        df.followingSatd = df.followingSatd.fillna('')
    if 'prevLineContent' in df.columns:
        df.prevLineContent = df.prevLineContent.fillna('')
    if 'nextLineContent' in df.columns:
        df.nextLineContent = df.nextLineContent.fillna('')
    
    # Find the following SATDs
    df = add_followingSatdCandidates(df)
    files_commits_matchingSatds = get_files_commits_matchingSatds(df)
    df = add_followingSatdByGreedy(df, files_commits_matchingSatds, 0.6, 0.2, 0, 0.2, 0.4) # these are the optimum weights we obtained through a grid search on a labeled dataset
    
    # merge the following SATDs
    df2 = df.copy()
    df2 = merge_followingSATDs_in_dataframe(df2, 'followingSatdByGreedy', True)
    print("Number of deleted SATDs after merging them:",len(df)-len(df2))
    print("Final number of SATDs:",len(df2))
    df2.to_csv(args.output)
    print("The extracted SATDs saved in", args.output)