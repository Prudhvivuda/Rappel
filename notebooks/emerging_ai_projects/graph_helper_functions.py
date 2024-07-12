import pandas as pd
import numpy as np
import collections
from operator import itemgetter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sqlalchemy as salc
import networkx as nx
import json
import matplotlib.colors as mcolors
import random
import uuid


def get_repos(repos, engine):

    repo_set = []
    repo_git_set = []
    for repo_git in repos:
        repo_query = salc.sql.text(f"""
                     SET SCHEMA 'augur_data';
                     SELECT 
                        b.repo_id,
                        b.repo_name,
                        b.repo_git
                    FROM
                        repo_groups a,
                        repo b
                    WHERE
                        a.repo_group_id = b.repo_group_id AND
                        b.repo_git = \'{repo_git}\'
            """)
        
        with engine.connect() as conn:
            t = conn.execute(repo_query)
        results = t.mappings().all()[0]
        repo_id = results['repo_id']
        repo_git = results['repo_git']
        repo_set.append(repo_id)
        repo_git_set.append(repo_git)
    return repo_set, repo_git_set



def get_issue_contributors(repo_set, engine):

    issue_contrib = pd.DataFrame()
    for repo_id in repo_set:
        repo_query = salc.sql.text(f"""
                    SET SCHEMA 'augur_data';
                    SELECT r.repo_id,
                    r.repo_git,
                    r.repo_name,
                    ie.cntrb_id,
                    ie.action,
                    i.issue_id,
                    i.created_at
                    FROM
                    repo r, issues i, issue_events ie
                     WHERE
                    i.repo_id = \'{repo_id}\' AND
                    i.repo_id = r.repo_id AND
                    i.issue_id = ie.issue_id AND
                    ie.action='closed'
            """)
        
        with engine.connect() as conn:
            df_current_repo = pd.read_sql_query(repo_query, conn)
        issue_contrib = pd.concat([issue_contrib, df_current_repo])

    issue_contrib = issue_contrib.reset_index()
    issue_contrib.drop("index", axis=1, inplace=True)
    issue_contrib.columns =['repo_id', 'repo_git', 'repo_name', 'cntrb_id', 'action', 'issue_id', 'created_at']
    return issue_contrib



def get_repos_outside(engine):

    issue_contrib = pd.DataFrame()
    repo_query = salc.sql.text(f"""
                    SET SCHEMA 'augur_data';
                    SELECT r.repo_name,
                    (CASE WHEN REGEXP_LIKE(repo_name, 'https://github.com/open-telemetry/opentelemetry-go|https://github.com/open-telemetry/opentelemetry-specification|https://github.com/open-telemetry/opentelemetry-collector') THEN true ELSE NULL
                    END) AS flag
                    FROM repo r
            """)
    
    with engine.connect() as conn:
        df_current_repo = pd.read_sql_query(repo_query, conn)
        
    print(df_current_repo)
    return issue_contrib



def get_pr_contributors(repo_set, engine):

    pr_contrib = pd.DataFrame()

    for repo_id in repo_set:
        repo_query = salc.sql.text(f"""
                    SET SCHEMA 'augur_data';
                    SELECT r.repo_id,
                    r.repo_git,
                    r.repo_name,
                    prm.cntrb_id,
                    prm.pull_request_id,
                    pr.pr_created_at
                    FROM
                    repo r, pull_request_meta prm, pull_requests pr
                    WHERE
                    prm.repo_id = \'{repo_id}\' AND
                    prm.repo_id = r.repo_id AND
                    prm.pull_request_id = pr.pull_request_id
            """)
        
        with engine.connect() as conn:
            df_current_repo = pd.read_sql_query(repo_query, conn)
        pr_contrib = pd.concat([pr_contrib, df_current_repo])

    pr_contrib = pr_contrib.reset_index()
    pr_contrib.drop("index", axis=1, inplace=True)
    pr_contrib.columns =['repo_id', 'repo_git', 'repo_name', 'cntrb_id', 'pull_request_id', 'pr_created_at']

    return pr_contrib



def get_commit_contributors(repo_set, engine):

    commit_contrib = pd.DataFrame()

    for repo_id in repo_set:
        repo_query = salc.sql.text(f"""
                    SET SCHEMA 'augur_data';
                    SELECT r.repo_id,
                    r.repo_git,
                    r.repo_name,
                    ca.cntrb_id,
                    c.cmt_id,
                    c.cmt_date_attempted
                    FROM
                    repo r, commits c, contributors_aliases ca
                    WHERE
                    c.repo_id = \'{repo_id}\' AND
                    c.repo_id = r.repo_id and
                    c.cmt_committer_email = ca.alias_email
            """)
        
        with engine.connect() as conn:
            df_current_repo = pd.read_sql_query(repo_query, conn)
        commit_contrib = pd.concat([commit_contrib, df_current_repo])

    commit_contrib = commit_contrib.reset_index()
    commit_contrib.drop("index", axis=1, inplace=True)
    commit_contrib.columns =['repo_id', 'repo_git', 'repo_name', 'cntrb_id', 'cmt_id', 'cmt_date_attempted']

    return commit_contrib



def get_prr_contributors(repo_set, engine):

    prr_contrib = pd.DataFrame()

    for repo_id in repo_set:
        repo_query = salc.sql.text(f"""
                    SET SCHEMA 'augur_data';
                    SELECT r.repo_id,
                    r.repo_git,
                    r.repo_name,
                    prr.cntrb_id,
                    prr.pull_request_id
                    FROM
                    repo r, pull_request_reviewers prr
                    WHERE
                    prr.repo_id = \'{repo_id}\' AND
                    prr.repo_id = r.repo_id
            """)
        
        with engine.connect() as conn:
            df_current_repo = pd.read_sql_query(repo_query, conn)
        prr_contrib = pd.concat([prr_contrib, df_current_repo])

    prr_contrib = prr_contrib.reset_index()
    prr_contrib.drop("index", axis=1, inplace=True)
    prr_contrib.columns = ['repo_id', 'repo_git', 'repo_name', 'cntrb_id', 'pull_request_id']

    return prr_contrib



def created_melted_dfs(df):

    df = df.groupby(['org_repo', 'cntrb_id']).size().unstack(fill_value=0)
    df = df.reset_index()

    df_melted = df.melt(['org_repo'], var_name = 'cntrb_id',value_name='number')
    df_melted = df_melted[df_melted[df_melted.columns[2]] != 0]

    return df_melted


def get_page_ranks(graph, top, repo_dict, scores):
    
    """
    This method takes in a graph, and returns the nodes ranked by page rank 
    graph: input graph
    top: top number of repos to subset after calculating the page rank
    known_repos: list of repository/community names known to us
    other_repos: list of repository/community names that we want to determine the importance of
    """
    
    pageranks = nx.pagerank(graph, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
    
    scores['page_rank'] = scores['repo'].map(pageranks)
    
    pr_dicts = collections.defaultdict(dict)
    
    for key in pageranks:  
        for repo_group in repo_dict:
            if key in repo_dict[repo_group]:
                pr_dicts[repo_group][key] = pageranks[key]
 
    top_repos = collections.defaultdict(dict)
    
    for pr_dict in pr_dicts:
        top_repos[str(pr_dict)] = dict(sorted(pr_dicts[pr_dict].items(), key = itemgetter(1), reverse = True)[:top])
            
    return top_repos, pageranks, scores


def get_betweenness_centrality(graph, top, repo_dict, scores):
    
    """
    This method takes in a graph, and returns the nodes ranked by betweenness centrality scores
    graph: input graph
    top: top number of repos to subset after calculating the betweenness centrality scores
    known_repos: list of repository/community names known to us
    other_repos: list of repository/community names that we want to determine the importance of
    """
    
    # Betweenness centrality measures the extent to which a node lies on paths between other nodes in the graph. 
    # Nodes with higher betweenness have more influence within a network. 
    # Thus repositories with higher centrality scores can thought to be influential in connection to other repositories in the network.
    
    bw_centrality = nx.betweenness_centrality(graph)

    scores['betweenness_centrality'] = scores['repo'].map(bw_centrality)
    
    bc_dicts = collections.defaultdict(dict)
    
    for key in bw_centrality:  
        for repo_group in repo_dict:
            if key in repo_dict[repo_group]:
                bc_dicts[repo_group][key] = bw_centrality[key]
 
    top_repos = collections.defaultdict(dict)
    
    for bc_dict in bc_dicts:
        top_repos[str(bc_dict)] = dict(sorted(bc_dicts[bc_dict].items(), key = itemgetter(1), reverse = True)[:top])
            
    return top_repos, bw_centrality, scores


def get_closeness_centrality(graph, top, repo_dict, scores):
    
    """
    This method takes in a graph, and returns the nodes ranked by closeness centrality scores
    graph: input graph
    top: top number of repos to subset after calculating the closeness centrality scores
    known_repos: list of repository/community names known to us
    other_repos: list of repository/community names that we want to determine the importance of
    """
    
    c_centrality = nx.closeness_centrality(graph)
    
    scores['closeness_centrality'] = scores['repo'].map(c_centrality)
    
    cc_dicts = collections.defaultdict(dict)
    
    for key in c_centrality:  
        for repo_group in repo_dict:
            if key in repo_dict[repo_group]:
                cc_dicts[repo_group][key] = c_centrality[key]
 
    top_repos = collections.defaultdict(dict)
    
    for cc_dict in cc_dicts:
        top_repos[str(cc_dict)] = dict(sorted(cc_dicts[cc_dict].items(), key = itemgetter(1), reverse = True)[:top])
            
    return top_repos, c_centrality, scores


def plot_graph(graph, repo_dict, size, title, weights=None, with_labels=True, alpha=None, edge_color='k'):
    
    """
    graph: the networkX graph that we want to plot
    known_repos: list of known repos for coloring
    other_repos: list of other repos for coloring
    size: can be either 'weighted', 'equal' or 'conditional'
    When size is 'weighted', the node sizes on the graph are based on the weights provided
    When size is 'equal', all nodes are the same size
    When size is 'conditional', nodes which belong to the weights array are larger than the rest of the nodes
    weights: this decides the size of the nodes in the 'weighted' and 'conditional' type sizes
    
    here we plot a networkx graph based on the provided parameters
    """
    patches = []

    nodes = graph.nodes()
    node_colors = []
    chosen_colors = random.sample(list(mcolors.TABLEAU_COLORS), len(repo_dict) + 1)
    repo_no = 0
    
    #assign random colors to each repo_group
    color_map = collections.defaultdict(str)   
    for repo_group in repo_dict:
        color_map[repo_group] = chosen_colors[repo_no]
        repo_no += 1
    color_map['Contributors'] = chosen_colors[-1]

    for n in nodes:
        color_assigned = False
        try:
            uuid.UUID(str(n))
            node_colors.append(color_map['Contributors'])
            continue
        except ValueError:
            for repo_group in repo_dict:                    
                if n in repo_dict[repo_group]:
                    node_colors.append(color_map[repo_group])
                    color_assigned = True
                    break
                else:
                    continue
        if color_assigned == False:
            node_colors.append("white")

    if size == 'weighted':
        node_sizes = [v * 10000 for v in weights.values()]
    elif size == 'conditional':
        node_sizes = [1000 if ns in weights else 50 for ns in nodes]
    elif size == 'equal':
        node_sizes = 300
    
    for color in color_map:
        patches.append(mpatches.Patch(color=color_map[color], label=color))
        
    fig, ax = plt.subplots(figsize=(15,15))

    font = {"color": "k", "fontsize": 15}
    
    ax.set_title(title, font)
    ax.legend(handles=patches)
    
    nx.draw_networkx(graph, node_color=node_colors, node_size=node_sizes, font_size=9, ax=ax, with_labels=with_labels, alpha=alpha, edge_color=edge_color)


def project_nodes_edges_contributions(df):
    
    """
    Using this function we represent data as a graph where the project repositories are represented by nodes 
    and the edges are shared contributions between those projects
    """
 
    # structure of `contributorGraph` =  
    # {  
    # `contributor1`: [(`repo1`, `contributions by the contributor1 in repo 1`)],  
    #  `contributor2`: [(`repo2`, `contributions by the contributor2 in repo 2` ), (`repo1`, `contributions by the contributor2 in repo 1`)]  
    # }

    contributorGraph = {}
    for i, row in df.iterrows():
        if row['cntrb_id'] not in contributorGraph:
            contributorGraph[row['cntrb_id']] = []
        if(row['total_contributions'] > 0):
            contributorGraph[row['cntrb_id']].append((row['org_repo'], row['total_contributions']))
            
    # `contributorGraph`  is a dictionary where each key is a contributor, 
    #  and the value is a list of repositories the contributor has contributed to and the number of contributions it has made.
    
    #  "shared connections" constitute of commits, PRs, issues* and PR reviews that are made by the same contributor.
    #  2 project repositories are "connected" if they have a "shared connection"** between them. 
    #  If they have a contributor who makes a commit, PR, issue or PR review in both the repositories, 
    #  they count as a shared contributor and the repositories are connected. 
    
    commonRepoContributionsByContributor = collections.defaultdict(int)
    for key in contributorGraph:
        if len(contributorGraph[key])-1 <= 0:
            continue
        for repoContributionIndex in range(len(contributorGraph[key])-1):
            commonRepoContributionsByContributor[(contributorGraph[key][repoContributionIndex][0], contributorGraph[key][repoContributionIndex+1][0])] += contributorGraph[key][repoContributionIndex][1]+contributorGraph[key][repoContributionIndex+1][1]

    # `commonRepoContributionsByContributor` is a nested dictionary consisting of dictionaries of repository pairs and their common contributions. 
    #  structure of `commonRepoContributionsByContributor` =  
    #  {  
    #  (`repo1, repo2`): `PRs by same authors in repo 1 and repo 2`,  
    #  (`repo2, repo4`): `PRs by same authors in repo 2 and repo 4`,  
    #  (`repo2, repo5`): `PRs by same authors in repo 2 and repo 5`,   
    #   }    
    
    res = []
    for key in commonRepoContributionsByContributor:
        res.append(tuple(str(k) for k in list(key)) + (commonRepoContributionsByContributor[key],))
        
    return res, commonRepoContributionsByContributor


def preprocess(df):
    
    """
    This method takes as input a dataframe with timestamp and contributor id data and converts timestamps from 
    strings to datetime objects and converts contributor ids from UUIDs to strings and abbreviates them.
    """
    
    # convert values in the created_at column from strings to datetime objects
    df_copy = df.copy()
    df_copy['created_at'] = pd.to_datetime(df['created_at'], utc='True')

    # shorten cntrb_ids
    df_copy['cntrb_id'] = df_copy['cntrb_id'].apply(lambda x: str(x).split('-')[0])
    # change dtype of cntrb_id that are None to string
    df_copy['cntrb_id'] = df_copy["cntrb_id"].replace(to_replace=[None], value='None')
    
    return df_copy



def calc_recency_weights(df):
    
    """
    This method takes as input a dataframe containing contribution data to repositories within a repository
    group with at least repo ids, contributor ids, and timestamp columns and outputs a copy of the dataframe
    with an additional column named `recency_weights` to weigh degree of participation of contributors to 
    projects by the recency of their contributionsTF-IDF of a contributor to a repository. Recency weights
    range from 0 to 1, the more recent a contribution, the closer its recency weight is to 1.
    """
    
    # make a copy of df and convert dtypes
    df_copy = preprocess(df)
    
    # get the most recent 'created_at' value
    most_recent = max(df_copy['created_at'])
    
    # recency weights are calculated as the negative square root of the difference, in number of days, between the most recent date and the 'created_at' date 
    # the more recent a contribution is, the more recency weight it has
    # we assign a recency weight of 1 to created_at values equal to the most recent date
    df_copy['recency_weights'] = df.apply(
        lambda row:  (-np.sqrt(float((most_recent - row.created_at).days))) if ((most_recent - row.created_at).days) !=0 else 1, axis=1
        )
    
    return df_copy


def calc_tfidf_weights(df):

    """
    This method takes as input a dataframe containing contribution data to repositories within a repository
    group with at least repo ids and contributor ids columns and outputs a copy of the dataframe with an 
    additional column named `tf-idf` to weigh degree of participation of contributors by TF-IDF. Our adapted 
    definition of TF-IDF measures the relevance of a contributor to a repository. We define a relevant contributor 
    as one that makes many contributions to a repository and few to others. TF-IDF weights range from 0 to 1 
    where the more relevant a contributor to a repository, the closer their TF-IDF weight is to 1.
    """

    # tf-idf is a measure of importance of a word to a document in corpus, 
    # adjusted by the overall frequency of that word in the corpus

    # we adapt this definition to measure contributor participation in a repo belonging
    # to repository group, adjusting for the the fact that some contributors contribute
    # more frequently to the repository group

    # make a copy of df and convert dtypes
    df_copy = preprocess(df)
    
    # count the number of repos each contributor has contributed to 
    repoContributionsByContributor = df_copy.groupby('cntrb_id')['repo_id'].count().to_dict()
    
    # count the number of contributions by repo and contributor 
    df_copy = df_copy.groupby(['repo_id', 'cntrb_id']).size().to_frame()
    df_copy = df_copy.reset_index()
    df_copy = df_copy.rename(columns={0:'num_cntrbs'})
    
    # count the total number of repos in the repo group
    totalRepos = df_copy['repo_id'].nunique()
    # count the number of contributions by repo
    contributonsByRepo = df_copy.groupby(['repo_id'])['num_cntrbs'].sum().to_dict()

    # tf is defined as the relative frequency of a individual's contributions to a repo i.e 
    # the sum of their contributions divided by the total number of contributions to the repo
    tf = df_copy.apply(lambda row: row.num_cntrbs / contributonsByRepo[row.repo_id], axis=1)

    # idf is defined as how 'unique' a contribution is, depending on how often or rarely a contributor 
    # contributes to the repository group, i.e the log of the total number of repos in a repository 
    # group divided by the number of repos each contributor has contributed to 
    idf = df_copy.apply(lambda row: np.log(totalRepos / repoContributionsByContributor[row.cntrb_id]), axis=1)
    
    df_copy['tf_idf'] = (tf * idf)
    
    return df_copy

