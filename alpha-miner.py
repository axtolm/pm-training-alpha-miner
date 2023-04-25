# -*- coding: utf-8 -*-
"""
Created on April 25, 2023
@author: Alexander Tolmachev axtolm@gmail.com
A streamlit web app "Process Mining training" - "Alpha Miner" module

"""
# packages
import streamlit as st
import pandas as pd
import graphviz
import itertools
import re
from copy import deepcopy

# default settings of the page
st.set_page_config(page_title="PM-training (Alpha Miner)", page_icon=":rocket:", 
                   layout= "wide", initial_sidebar_state="expanded")
# hide right menu and logo at the bottom 
hide_streamlit_style = """
                       <style>
                       #MainMenu {visibility: unhidden;}
                       footer {visibility: hidden;}
                       </style>
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)              

def main():
    # =============================================================================
    LNG = 'en'                  # interface language
    md_text = get_dict_text()   # dict with markdown texts
    # =============================================================================
    # left panel      
    # =============================================================================
    page = st.sidebar.radio('**Alpha Miner**', 
                            ['Alpha Algorithm']) 
    graph_orientation = st.sidebar.radio('**Graph orientation (Left → Right or Top → Bottom)**',['LR','TB'],index = 0, horizontal = True)
    st.sidebar.markdown('---')
    st.sidebar.markdown(md_text['left_block_author_refs',LNG])                    
    # =============================================================================   
    # central panel
    # =============================================================================   
    st.markdown('##### Process Mining training. Alpha Miner (Bottom-Up Process Discovery)')   
    # =============================================================================
    # common block #0 - Context and Definitions
    with st.expander("Alpha Miner Contents and Definitions", expanded = False):
        # Show Definitions 
        st.markdown(md_text['cb_recall_dfg_def',LNG]) 
        st.info(md_text['cb_alpha_algo_definition',LNG])
        st.markdown(md_text['cb_contents_1',LNG]) 
        st.info(md_text['cb_dfg_definition',LNG])
        st.markdown(md_text['cb_contents_2',LNG])
        st.info(md_text['cb_discovery_dfg_definition',LNG])
        st.markdown(md_text['cb_contents_3',LNG])
        st.info(md_text['cb_dfg_footprint_definition',LNG])
        st.markdown(md_text['cb_contents_4',LNG])
        st.info(md_text['cb_accepting_petri_net_definition',LNG])
        st.markdown(md_text['cb_contents_5',LNG])
        st.info(md_text['cb_labeled_petri_net_definition',LNG])
    # =============================================================================
    # common block #1 - event log selection 
    st.markdown(md_text['common_block',LNG])    
    with st.expander("Select an event log for training", expanded = True):
        # show the default simple event logs
        st.markdown(md_text['log_list',LNG])
        
        col1_log, col2_log = st.columns(2)
        # selection radio buttons
        st_radio_select_log = col1_log.radio('Choose one of the default event logs', 
                                     ('L1','L2','L3','L4','L5','L6','L7','L8'), 
                                       index = 0, horizontal = True)
        # show checkbox
        col2_log.write(''); col2_log.write('')
        st_check_edit_log = col2_log.checkbox('Modify selected event log',value = False)
        
        if st_check_edit_log:
            selected_log = st.text_input('Edit your event log as a string', value = get_default_event_log(st_radio_select_log))
            st.markdown(md_text['user_log_format_requirements',LNG])
        else:
            selected_log = get_default_event_log(st_radio_select_log)

    # =============================================================================    
    # common block #2 - check the selected event log 
    with st.expander("Check the selected event log in the table format", expanded = True): 
        try:
            df_log = get_df_log(selected_log)
            if len(df_log)==0: raise Exception ('Error! Check your input data')   # simple error test
            st.write(df_log)   # show DataFrame
        except Exception as ex_msg: st.warning(ex_msg)        
    
    # =========================================================================
    # Exercise #1 - Alpha Algorithm
    # =========================================================================
    if page == 'Alpha Algorithm':  
        st.markdown('##### Constructing an Accepting Petri-Net based on the simple Event Log using the Alpha Algorithm')  
        with st.form('Applying the Alpha Algorithm'):
            st_submitt_start_alpha_algorithm = st.form_submit_button('Start the Alpha Algorithm')
            if st_submitt_start_alpha_algorithm:
                # =========================================================================
                # executive python code
                # =========================================================================
                df_log['trace'] = 'I' + df_log['trace'] + 'O'  # add start & end
                traces_list = list(df_log['trace'])
                qty_list = list(df_log['qty'])
                
                DFG_nodes, DFG_arcs = get_DFG (list(df_log['trace']), list(df_log['qty'])) # get DFG nodes & arcs
                vDFG = get_vDFG(DFG_arcs, DFG_nodes, graph_orientation,'I','O')   # construct DFG as graphviz object
                df_footprint, dict_footprint = get_footprint(list(DFG_arcs['pair']), list(DFG_arcs['qty']), 'I','O')
                
                T, P, F, T_label_dict = alpha_miner(traces_list, qty_list, 'I', 'O')
                vPN = get_vPetriNet(list(T['label']), list(P['place']), F, 'I', 'O', graph_orientation)                
                
                P['place(view)'] = [[s.replace('\n',' ')] for s in list(P['place'])]
                F_view = pd.DataFrame({'arc':[(str(s)).replace('\\n',' ').replace("'","").replace('(','').replace(')','').split(',') for s in F]})
                T['label_view'] = [[s] for s in list(T['label'])]
                # =========================================================================
                # web-page             
                # Preliminaries. Construct the DFG and get its footprint
                st.markdown(md_text['p1_preliminaries_get_DFG_and_footpint',LNG])
                col1, col2 = st.columns(2)
                col1.markdown('DFG footprint')
                col1.write(df_footprint)
                col2.markdown('Directly-Follows Graph')
                col2.image(vDFG)
                
                # Applying the Alpha Algorithm
                st.markdown(md_text['p1_applying_alpha_algorithm_title',LNG])                
                
                col1,col2,col3,col4 = st.columns([1,1,1,1])
                
                col1.markdown(md_text['p1_alpha_algorithm_step_1_2',LNG])
                col1.write(P[P.candidate !=''][['candidate']])   # the maximal candidates (Sel)
                
                col2.markdown(md_text['p1_alpha_algorithm_step_3',LNG])
                col2.write(P[['place(view)']])    # the set of places P
                
                col3.markdown(md_text['p1_alpha_algorithm_step_4_6',LNG])
                col3.write(T[['label_view','transition']])    # the set of transitions T and labeling function l
                
                col4.markdown(md_text['p1_alpha_algorithm_step_5',LNG])
                col4.write(F_view)    # the set of arcs F
                
                st.markdown(md_text['p1_alpha_algorithm_step_7',LNG])    # the initial and final marking
                st.markdown(md_text['p1_alpha_algorithm_step_8',LNG])    # the discovered accepting Petri net
                st.image(vPN)
                
                with st.expander('Features and limitations of the Alpha Algorithm', expanded = False):
                    st.markdown(md_text['p1_alpha_algorithm_features_limitations_1',LNG])    
                    st.info(md_text['p1_alpha_algorithm_features_limitations_2',LNG])
                    st.markdown(md_text['p1_alpha_algorithm_features_limitations_3',LNG])
                st.success(md_text['p1_summary',LNG], icon="✅")

# =============================================================================
# Service functions
# =============================================================================
# Common block 
# =============================================================================
def get_df_log(str_log):
    '''
    Event log transformation from str ('[<acd>45, <bce>42]') to pandas.DataFrame (columns = ['trace','qty'])
    '''
    return pd.DataFrame({'trace':re.findall('[a-z]+', str_log),'qty':[int(s) for s in re.findall('[0-9]+', str_log)]})
# =============================================================================
# Exercise #1 - Alpha Algorithm
# =============================================================================
def get_DFG (traces_list, qty_list):
    '''
    Computing the DFG nodes - (activity, frequency), and
    the DFG arcs - ((activity,activity), frequency)

    Parameters
    ----------
    traces_list : list
        list of traces (e.g. traces_list = ['acd','bce'])
    qty_list : list
        list of frequencies of traces (e.g. qty_list = [45,42] )
    Returns
    -------
    DFG_nodes_agg : pandas.DataFrame
        2 columns: 'act' - activities, 'qty' - their frequencies in the event log
    DFG_arcs_agg : pandas.DataFrame
        2 columns: 'pair' - arcs, 'qty' - their frequencies in the event log
    
    Example
    -------
    DFG_nodes, DFG_arcs = get_DFG (list(df_log['trace']), list(df_log['qty']))
    '''   
    # create internal pd.DataFrame based on input data
    L = pd.DataFrame({'trace_full':traces_list, 'qty': qty_list})
    # =============================================================================
    # NODES (ACTIVITIES)
    # function get_activities returns pd with pairs (activity, frequency) for one trace ('abc') with frequency (20)
    get_activities = lambda trace, qty: pd.DataFrame({'act':list(trace),'qty':qty})
    # compute DFG nodes as pd for all traces using the function get_activities
    L['nodes'] = list(map(lambda trace, qty: get_activities(trace, qty), L['trace_full'], L['qty']))
    # merge and aggregate all DFG nodes with sorting in descending frequency
    DFG_nodes = pd.concat([L.iloc[i]['nodes'] for i in list(L.index)])
    DFG_nodes_agg = pd.DataFrame(DFG_nodes.groupby(['act']).sum()).sort_values(by=['qty'], ascending=False).reset_index()
    # =============================================================================
    # EDGES (ARCS)
    # function get_pairs returns pd with pairs (arc, frequency) for one trace ('abc') with frequency (20)
    get_pairs = lambda trace, qty: pd.DataFrame({'pair':list(map(lambda s1, s2: s1 + s2, trace[:-1], trace[1:])),'qty':qty})
    # compute DFG edges (arcs) as pd for all traces using the function get_pairs
    L['arcs'] = list(map(lambda trace, qty: get_pairs(trace, qty), L['trace_full'], L['qty']))
    # merge and aggregate all DFG nodes with sorting in descending frequency
    DFG_arcs = pd.concat([L.iloc[i]['arcs'] for i in list(L.index)])
    DFG_arcs_agg = pd.DataFrame(DFG_arcs.groupby(['pair']).sum()).sort_values(by=['qty'], ascending=False).reset_index()
    
    return DFG_nodes_agg, DFG_arcs_agg

def get_vDFG(DFG_arcs, DFG_nodes, DFG_orientation,S,E):
    '''
    Creating the DFG by Graphviz 

    Parameters
    ----------
    DFG_arcs : pandas.DataFrame
        table with arcs (2 columns: 'pair' - arcs, 'qty' - their frequencies in the event log)
    DFG_nodes : pandas.DataFrame
        table with nodes (2 columns: 'act' - activities, 'qty' - their frequencies in the event log)
    DFG_orientation : str
        'LR' or 'TB' (left -> right or top -> bottom)
    S : str
        symbol for the artificial activity 'Start'- 'I'
    E : str
        symbol for the artificial activity 'End'- 'O'

    Returns
    -------
    <class 'bytes'>
        vDFG.pipe() for vizualization by st.image(vDFG)
    Example
    -------
    vDFG = get_vDFG(DFG_arcs, DFG_nodes, dfg_orientation,'I','O')
    '''
    # init graph
    vDFG = graphviz.Digraph('finite_state_machine')
    vDFG.attr(rankdir = DFG_orientation, size = '1000,1000') 
    vDFG.format = 'png'
    # DFG NODES 
    for i in DFG_nodes.index:
        # start or end - double circles
        if (DFG_nodes['act'].loc[i] == S)|(DFG_nodes['act'].loc[i] == E):
            vDFG.attr('node', shape='doublecircle')
            vDFG.node(DFG_nodes['act'].loc[i], label = DFG_nodes['act'].loc[i])
        else:
            vDFG.attr('node', shape='circle')
            vDFG.node(DFG_nodes['act'].loc[i], label = DFG_nodes['act'].loc[i])
    # DFG EDGES 
    for j in DFG_arcs.index:
        vDFG.edge(DFG_arcs['pair'].loc[j][0], DFG_arcs['pair'].loc[j][1], label = str(DFG_arcs['qty'].loc[j]))
    return vDFG.pipe()

def get_footprint(in_pairs,in_qty_list,S,E):
    '''
    Computing the alternative DFG representations:
    - matrix with frequencis of arcs, 
    - footprint with relations between activities.

    Parameters
    ----------
    in_pairs : list
        list of arcs, i.e. in_pairs = ['Sa','eE']
    in_qty_list : list
        list of frequencies of arcs  (e.g. in_qty_list = [45,42] )
    S : str
        Start symbol (e.g. S = 'I')
    E : str
        Start symbol (e.g. E = 'O')

    Returns
    -------
    df_footprint : pandas.DataFrame
        table with relations between activities (columns and rows - [I,a,b,...,O])
    dict_footprint : dict
        footprint as a dictionary (i.e. dict_footprint['ab'] can returns string '||')

    '''
    # function to get relation between activities in pair
    get_rel = lambda pair,pairs: "||" if (''.join(list(pair)[::-1])) in pairs else "→"
    # get relation and qty lists
    rel = list(map(lambda ss: get_rel(ss, in_pairs), in_pairs))
    # get reverse pairs/relations
    in_pairs_reverse = list(map(lambda ss: ''.join(list(ss)[::-1]), in_pairs))
    rel_reverse = list(map(lambda ss: "||" if (ss == "||") else "←", rel))
    # get full df with index = pairs & rel - relation plus transform it to dict
    arcs_full = pd.DataFrame({'rel':rel+rel_reverse}, index = in_pairs+in_pairs_reverse)
    arcs_dict = arcs_full['rel'].to_dict()
    # get sorted activity list
    act_nodes = list(set(list(''.join(in_pairs).replace(S,'').replace(E,'')))) # without S, E
    act_sorted = [S]+sorted(act_nodes)+[E]
    # get the result df
    df_footprint = pd.DataFrame(index = act_sorted)
    for col in act_sorted:
        df_footprint[col] = list(map(lambda ss: 
                                     arcs_dict[ss+col] if (ss+col) in list(arcs_full.index) else '#', act_sorted))
    # get footprint dict
    all_arcs = [''.join(list(s)) for s in list(itertools.product(act_sorted,act_sorted))]
    other_arcs = list(set(all_arcs) - set(arcs_full.index))   # dict with ''#''
    dict_footprint = {ss: '#' for ss in other_arcs} | arcs_dict  # merge two dicts
    return df_footprint, dict_footprint

def split_nodes(in_nodes, arcs_connect_all): 
    '''
    Split the input set of activities into a list of subsets, 
    where each subset is a set of actvities with only # relations.  

    Parameters
    ----------
    in_nodes : set
        Set of activities for splitting,
        e.g. {'e', 'f', 'c', 'd'}.
    arcs_connect_all : set
        Set of all arcs in the DFG (non # relations),
        e.g. {'bd', 'ec', 'eg', 'bf',...}.
    Returns
    -------
    nodes : list of sets
        List of sets, where each set is the set of actvities with # relations,  
        e.g. [{'e', 'f'}, {'c', 'f'}, {'d', 'e'}, {'c', 'd'}].

    '''
    nodes = [deepcopy(in_nodes)]   # an initial list with one input set
    # get all posible pairs based on activities from the input set
    # permutations(p[,r]): r-length tuples, all possible orderings, no repeated elements ('aa' pairs are not required)
    arcs = {''.join(list(s)) for s in list(itertools.permutations(in_nodes,2))}  
    # get not # arcs from all posible pairs 'arcs'
    arcs_connected = arcs.intersection(arcs_connect_all)
    # merge pairs 'cd' and 'dc' into one 'cd'
    arcs_connected_ordered = {''.join(sorted(s)) for s in arcs_connected} 
    # split initial set into subsets with non-connected nodes
    for carc in arcs_connected_ordered: 
        nodes_split = []
        for node in nodes: 
            nodes_split.append({char if char != carc[0] else 'NaN' for char in node}.difference({'NaN'}))
            nodes_split.append({char if char != carc[1] else 'NaN' for char in node}.difference({'NaN'}))
        nodes = deepcopy(nodes_split)
    return nodes

def get_max_candidates(nodes_list, arcs_list, dict_fp):
    '''
    Get all maximal candidates for the Petri net places

    Parameters
    ----------
    nodes_list : list
        list of the DFG nodes,
        e.g. ['I','O','g','b','d','f','a','c','e']
    arcs_list : list
        list of the DFG arcs,
        e.g. ['gO','Ib','Ia','bf','dg','fd','ae','cg','ec','ac','bd','ce','df','eg','fg']
    dict_fp : dict
        full dictionary of relationships between DFG nodes (size NxN, N - qty of nodes),
        e.g. {'ed':'#',...,'gO':'→',...,'fd':'||',...,'bI':'←',...}
    Returns
    -------
    A1A2_full : list of lists of sets
        all maximal candidates,
        e.g. [[{'a'},{'e'}], [{'a'},{'c'}], [{'e','f'},{'g'}], [{'c','f'},{'g'}],
         [{'d','e'},{'g'}], [{'c','d'},{'g'}], [{'g'},{'O'}], [{'I'},{'a','b'}],
         [{'b'},{'f'}],[{'b'}, {'d'}]]

    '''
    # get the set of all nodes from the input - nodes_list
    nodes_all = set(nodes_list)
    # get the set of all nodes that are self connected (loops of length 1) - a||a
    nodes_self_connected =  {x if dict_fp[x+x]=='||' else 'NaN' for x in nodes_all}.difference({'NaN'}) 
    
    # get the set of all arcs from the input - arcs_list
    arcs_connect_all = set(arcs_list)  # relations are || or -> or <-
    # get the set of all possible arcs for the nodes_all  - product(p,q) - cartesian product, equivalent to a nested for-loop
    arcs_all  = set(''.join(list(s)) for s in list(itertools.product(nodes_all, nodes_all)))
    # get all arcs that connect activities directly (relation is '→')
    arcs_connect_direct = {x if (dict_fp[x]=='→') else 'NaN' for x in arcs_all}.difference({'NaN'}) # only → ('||' - exclude, '<-' are not necessary)

    # lookig for the maximal candidates   
    A1A2_full = []    # list of lists of sets for maximal candidates
    for a in nodes_all: # loop for all nodes, including I and O
        # select activities connected with 'a' on the right (xi: a-> xi)     
        A1A2 = [{a},{x if dict_fp[a+x] == '→' else 'NaN' for x in nodes_all}.difference({'NaN'})]    
        A1A2[1] = A1A2[1].difference(nodes_self_connected)     # remove self_connected a||a, i.e. ...aa...
        # select activities connected with 'xi' on the left (yi: yi->xi)
        for x in A1A2[1]:
            A1A2[0] = A1A2[0].union({y if dict_fp[y+x] == '→' else 'NaN' for y in nodes_all}.difference({'NaN'}))
        A1A2[0] = A1A2[0].difference(nodes_self_connected)     # remove self_connected a||a, i.e. ...aa...        
        # Check that each left activity is connected to all right activities, if not - remove it from the left ones
        for x in A1A2[0]:  
            x_arcs = {x+y for y in A1A2[1]}
            if len(x_arcs - arcs_connect_direct)>0: # here we detect 'a-b-a' relations (||) and exсlude them (loops of length 2)
                A1A2[0] = A1A2[0] - set(x)
        # check '#' in A1 (left) and A2 (right), if not # - split set
        left_nodes = split_nodes(A1A2[0], arcs_connect_all)
        if sum([len(x) for x in left_nodes])==0: continue
        right_nodes = split_nodes(A1A2[1], arcs_connect_all)
        if sum([len(x) for x in right_nodes])==0: continue    
        # get all candidates for left and right groups of nodes
        A1A2_candidates = [[a_left,a_right] for a_left in left_nodes for a_right in right_nodes]
        # check candidates with the general list
        for candidate in A1A2_candidates: 
            if not(candidate in A1A2_full):  # if the candidate is not in the general list: flag = True
                flag = True
                for selected in A1A2_full:   # check that candidate is maximal - if the candidate is a subset of some element of the general list: flag = False (not max)
                    if (candidate[0].issubset(selected[0]))&(candidate[1].issubset(selected[1])): flag = False
                    if (selected[0].issubset(candidate[0]))&(selected[1].issubset(candidate[1])): 
                        A1A2_full.remove(selected)    # if some element of the general list is a subset of candidate: delete the exiting element from the general list
                if flag: A1A2_full.append(candidate)  # add the candidate to the general list 
    return A1A2_full

def alpha_miner(traces_list, qty_list, a_start = 'I', a_end = 'O'):
    
    DFG_nodes, DFG_arcs = get_DFG(traces_list, qty_list)
    df_footprint, dict_footprint = get_footprint(list(DFG_arcs['pair']), list(DFG_arcs['qty']), a_start, a_end)
    max_candidates = get_max_candidates(list(DFG_nodes['act']), list(DFG_arcs['pair']), dict_footprint)
    # transitions   
    T = pd.DataFrame({'label':[a_start]+sorted(list(''.join(list(DFG_nodes['act'])).replace(a_start,'').replace(a_end,'')))+[a_end], 
                      'transition': [f'T{i}' for i in range(len(DFG_nodes['act']))]})
    T_label_dict = pd.DataFrame({'label':list(T['label'])}, index = list(T['transition']))['label'].to_dict()
    # places
    P = pd.DataFrame({'candidate': max_candidates,
                      'place':['P\n'+''.join(s[0]) +'_'+ ''.join(s[1]) for s in max_candidates]
                      }).sort_values(by='place').reset_index(drop=True)
    # arcs
    F = set()
    for i in P.index:
        for A1 in P.iloc[i]['candidate'][0]: F = F | {(A1, P.iloc[i]['place'])}
        for A2 in P.iloc[i]['candidate'][1]: F = F | {(P.iloc[i]['place'], A2)}
    # add Start(I) and End(O) to places and arcs
    P = pd.concat([pd.DataFrame({'candidate': [''],'place':['P\n'+a_start]}), P, pd.DataFrame({'candidate': [''],'place':['P\n'+a_end]})])        
    F = list(F | {('P\n'+a_start,a_start)} | {(a_end,'P\n'+a_end)})
    return T, P, F, T_label_dict 
    
def get_vPetriNet(T,P,F,a_start='I',a_end='O',PN_orientation = 'LR'):
    '''
    Create the Petri net graph by Graphviz

    Parameters
    ----------
    T : list
        List of transition labels (activities including start and end),
        e.g. T = ['I','a','b','c','d','e','O'].
    P : list
        List of places, 
        e.g. P = ['P\nI','P\nI_a','P\na_bd','P\na_dc','P\nbd_e','P\ncd_e','P\ne_O','P\nO'],
        '\n' - visualization in two lines, 'a_bd' - label: left transition is 'a', right - 'b' and 'd'. 
    F : list 
        List of arcs in tuples: '(a,b)' is arc from 'a' to 'b' 
    a_start : str, optional
        Start activity designation. The default is 'I'.
    a_end : str, optional
        End activity designation. The default is 'O'.
    PN_orientation : str, optional
        'LR' or 'TB' (left -> right or top -> bottom). The default is 'LR'.

    Returns
    -------
    <class 'bytes'>
        vPetriNet.pipe() for vizualization by st.image(vPetriNet)

    '''
    # init graph
    vPetriNet = graphviz.Digraph('TrafficLights')
    vPetriNet.attr(rankdir = PN_orientation, size='1000,1000') 
    vPetriNet.format = 'png'
    # add transitions
    for t in T: 
        # highlight the start and end
        if t in [a_start]: vPetriNet.attr('node', width='0.3', fillcolor='#66FF66')
        elif t in [a_end]: vPetriNet.attr('node', width='0.3', fillcolor='#FF7C80')
        else: vPetriNet.attr('node', width='0.5', fillcolor='white')
        vPetriNet.attr('node', shape='box', fixedsize='true', style='filled', fontname = 'arial black', fontsize='12', fontcolor='black')            
        vPetriNet.node(t)  
    # add places
    for p in P:
        # highlight the start and end
        if p in ['P\n'+a_start]: vPetriNet.attr('node', shape='circle', fillcolor='#66FF66') 
        elif p in ['P\n'+a_end]: vPetriNet.attr('node', shape='doublecircle', fillcolor='#FF7C80') 
        else: vPetriNet.attr('node', shape='circle', fillcolor='white')            
        vPetriNet.attr('node', fixedsize='true', width='0.6', style='filled',  fontname = 'arial', fontsize='10', fontcolor='black') 
        vPetriNet.node(p)    
    # add arcs
    for f in F: 
        vPetriNet.edge(f[0],f[1])
    return vPetriNet.pipe()    

# =============================================================================
# Special function to get texts in markdown format
# =============================================================================
def get_default_event_log(L):
    if   L == 'L1': return '[<abce>50,<acbe>40,<abcdbce>30,<acbdbce>20,<abcdcbe>10,<acbdcbdbce>10]'
    elif L == 'L2': return '[<aceg>2, <aecg>3,<bdfg>2,<bfdg>4]'
    elif L == 'L3': return '[<acd>45, <bce>42]'
    elif L == 'L4': return '[<abab>5, <ac>2]'
    elif L == 'L5': return '[<abce>10,<acbe>5,<ade>1]' 
    elif L == 'L6': return '[<ab>35, <ba>15]'
    elif L == 'L7': return '[<a>10, <ab>8,<acb>6,<accb>3,<acccb>1]'
    elif L == 'L8': return '[<abef>2,<abecdbf>3,<abcedbf>2,<abcdebf>4,<aebcdbf>3]'
    else: return '[<abce>50,<acbe>40,<abcdbce>30,<acbdbce>20,<abcdcbe>10,<acbdcbdbce>10]'
    
def get_dict_text():
    dict_text = dict()
    # left block
    dict_text['left_block_author_refs','en'] = (''' 
           A streamlit web app "Process Mining training"      
           "Alpha Miner" module      
           v1.0.1 (2023)     
                
           Developed by Alexander Tolmachev (axtolm@gmail.com)    
           
           References   
           1. van der Aalst, W.M.P.: Foundations of Process Discovery. 
           In: van der Aalst, W.M.P., Carmona, J. (eds.) PMSS 2022. 
           LNBIP, vol. 448, pp. 37–75. Springer, Cham (2022). 
           [link](https://doi.org/10.1007/978-3-031-08848-3_2)
           ''')
    # common block
    dict_text['cb_recall_dfg_def','en'] = (''' 
           We will use **the definition of the Alpha Algorithm** from [1].    
           ''')       
    dict_text['cb_alpha_algo_definition','en'] = ('''
           *The alpha algorithm $disc_{alpha} \in B(U^{*}_{act}) \\to U_{AN}$ returns an accepting Petri net*
           *$disc_{alpha}(L)$ for any event log $L \in B(U^{*}_{act})$*.
           *Let $A = act(L)$ and $fp(L) = fp(disc_{DFG}(L))$ the footprint of event log $L$.*
           *This allows us to write $a_1 \\to_L a_2$ if $fp(L)((a_1,a_2)) = \#$ for any $a_1,a_2 \in A' = A \cup \{I,O\}$ 
           (we use dummy start ($I$) and end ($O$) activities).*    
           1. $Cnd = \{(A_1,A_2) | A_1 \\subseteq A' \\land A_1 \\neq \oslash \\land A_2 \\subseteq A' \\land A_2 \\neq \oslash 
                       \\land \\forall _{a_1 \in A_1} \\forall _{a_2 \\in A_2} a_1 \\to _L a_2 
                       \\land \\forall _{a_1,a_2 \\in A_1} a_1 \#_L a_2 \\land \\forall _{a_1,a_2 \in A_2} a_1 \#_L a_2\}$ 
                       *are the candidate places*,    
           2. $Sel = \{(A_1,A_2) \\in Cnd | \\forall _{(A'_1,A'_2) \\in Cnd} A_1 \\subseteq A'_1 \\land A_2 \\subseteq A'_2
                       \\implies (A_1,A_2) = (A'_1,A'_2))\}$ *are the selected maximal places*,     
           3. $P = \{p_{(A_1,A_2)} | (A_1,A_2) \\in Sel\} \\cup \{p_I,p_O\}$ *is the set of all places*,    
           4. $T = \{t_a | a \\in A' \}$ *is the set of transitions*,    
           5. $F = \{(t_a, p_{(A_1,A_2)}) | (A_1,A_2) \\in Sel \\land a \\in A_1\} \\cup
                    \{(p_{(A_1,A_2)},t_a) | (A_1,A_2) \\in Sel \\land a \\in A_2\} \\cup
                    \{(p_I,t_I),(t_O,p_O)\}$ *is the set of arcs*,    
           6. $l = \{(t_a,a) | a \\in A\}$ *is the labeling function*,    
           7. $M_{init} = [p_I]$ *is the initial marking*, $M_{final} = [p_O]$ *is the final marking, and*    
           8. $disc_{alpha}(L) = ((P,T,F,l),M_{init},M_{final})$ *is the discovered accepting Petri net*.
                                                 
           ''') 
    dict_text['cb_contents_1','en'] = ('''              
               In all our exercises, we can use one of the pre-installed simple event logs or create our own by modifying one of the pre-installed ones. 
               
               **Exercise #1 (Alpha Algorithm)**.     
               Based on the chosen event log, we will create a Directly-Follows Graph 
               (by the Baseline Discovery Algorithm) and calculate the DFG footprint. 
               Using the DFG footprint, we will perform the Alpha Algorithm step by step. 
               As a result, we will represent the discovered process model in the form 
               of a table and a Petri net graph.    
                   
               **There are several terms in the definition of the Alpha algorithm**:   
               1. **Directly-Follows Graph (DFG)**    
               ''')                
    dict_text['cb_dfg_definition','en'] = ('''
           *A Directly-Follows Graph (DFG) is a pair $G=(A,F)$ where* 
           - *$A \subseteq U_{act}$ is a set of activities, and* 
           - *$F \in B((A × A) \cup (\{I\} × A) \cup (A × \{O\}) \cup (\{I\} × \{O\}))$ is a multiset of arcs.*\n 
           *$I$ is the start node and $O$ is the end node $(\{I,O\} \cap U_{act} = \oslash)$*. 
           *$U_{G} \subseteq U_{M}$ is the set of all DFGs*.                       
           ''') 
    dict_text['cb_contents_2','en'] = ('''              
               2. **The Baseline Discovery Algorithm** is used to construct the Directly-Follows Graph (DFG).    
               ''')            
    dict_text['cb_discovery_dfg_definition','en'] = ('''
            *Let $L \in B(U_{act}^*)$ be an event log. $disc_{DFG}(L) = (A,F)$ is the DFG based on $L$ with:*
            - *$A = \{a \in \sigma | \sigma \in L\}$, and*
            - *$F = [(\sigma_{i},\sigma_{i+1}) | \sigma \in L\'  \wedge 1 \leq i < |\sigma|]$* 
                    *with $L\' = [<I> ^{.} \sigma ^{.} <O> | \sigma \in L]$*                      
               ''')                 
    dict_text['cb_contents_3','en'] = ('''              
                   3. **The DFG footprint** captures the relations between activities.    
                   ''')                                                  
    dict_text['cb_dfg_footprint_definition','en'] = ('''
           *Let $G = (A, F) \in U_G$ be a DFG.*     
           *$G$ defines a footprint $fp(G) \in (A'$x$A') → \{→,←,\|,\#\}$ such that $A' = A \cup \{I,O\}$ and for any $(a_1, a_2) \in A'$x$A'$:*   
            - *$fp(G)((a_1,a_2)) =$ "$→$" if $(a_1,a_2) \in F$ and $(a_2,a_1)$ $\\notin$ $F$,*   
            - *$fp(G)((a_1,a_2)) =$ "$←$" if $(a_1,a_2) \\notin F$ and $(a_2,a_1) \in F$,*    
            - *$fp(G)((a_1,a_2)) =$ "$\|$" if $(a_1,a_2) \in F$ and $(a_2,a_1) \in F$, and*     
            - *$fp(G)((a_1,a_2)) =$ "$\#$" if $(a_1,a_2) \\notin F$ and $(a_2,a_1) \\notin F$*.    
           ''')    
    dict_text['cb_contents_4','en'] = ('''              
                   4. As a result, we need to get **an accepting Petri net**.    
                   ''')                  
    dict_text['cb_accepting_petri_net_definition','en'] = ('''
           *An accepting Petri net is a triplet $AN = (N, M_{init}, M_{final})$ where $N = (P, T, F, l)$ 
           is a labeled Petri net, $M_{init} \\in B(P)$ is the initial marking, and $M_{final} \\in B(P)$
           is the final marking. $U_{AN} \\subseteq U_M$ is the set of accepting Petri nets.*
           ''')     
    dict_text['cb_contents_5','en'] = ('''              
                   5. The previous definition mentions **a labeled Petri net**.    
                   ''')                  
    dict_text['cb_labeled_petri_net_definition','en'] = ('''
           *A labeled Petri net is a tuple $N = (P, T, F, l)$ with $P$ the set of places, 
           $T$ the set of transitions, $P \\cap T = \\oslash$, $F \\subseteq (P \\times T) \\cup (T \\times P)$ 
           the flow relation, and $l \\in T \\not\\to U_{act}$ a labeling function. 
           We write $l(t) = \\tau$ if $t \\in T \\backslash dom(l)$ (i.e., $t$ is a silent transition that cannot be observed).*
           ''')  
     
    dict_text['common_block','en'] = ('''
           Choose an event log or make your own by clicking the `Modify` checkbox and editing the selected event log. 
           View the event log in a table format and click the button to analyze the algorithm step by step.                           
           ''')     
    dict_text['log_list','en'] = ('''
           $L_1 = [<a,b,c,e>^{50},<a,c,b,e>^{40},<a,b,c,d,b,c,e>^{30},<a,c,b,d,b,c,e>^{20},<a,b,c,d,c,b,e>^{10},<a,c,b,d,c,b,d,b,c,e>^{10}]$,     
           $L_2 = [<a,c,e,g>^{2},<a,e,c,g>^{3},<b,d,f,g>^{2},<b,f,d,g>^{4}]$,     $L_3 = [<a,c,d>^{45},<b,c,e>^{42}]$,   $L_4 = [<a,b,a,b>^{5},<a,c>^{2}]$,       
           $L_5 = [<a,b,c,e>^{10},<a,c,b,e>^{5},<a,d,e>^{1}]$,   $L_6 = [<a,b>^{35},<b,a>^{15}]$,   $L_7 = [<a>^{10},<a,b>^{8},<a,c,b>^{6},<a,c,c,b>^{6},<a,c,c,c,b>^{6}]$,     
           $L_8 = [<a,b,e,f>^{2},<a,b,e,c,d,b,f>^{3},<a,b,c,e,d,b,f>^{2},<a,b,c,d,e,b,f>^{4},<a,e,b,c,d,b,f>^{3}]$
           ''') 
    dict_text['user_log_format_requirements','en'] = ('''
           Use the traditional format of a simple event log: 
           `[<acd>45, <bce>42]`, where `<acd>` is the trace, 
           and `45` is the number of times this trace appears in the event log
           ''')   
    # page 1 - Alpha Algorithm  
    
    dict_text['p1_preliminaries_get_DFG_and_footpint','en'] = ('''
           **Preliminaries. Construct the DFG and get its footprint** (for details, see the module "Directly-Follows Graph").
           ''')
    dict_text['p1_applying_alpha_algorithm_title','en'] = ('''
           **Applying the Alpha Algorithm**
           ''') 
    dict_text['p1_alpha_algorithm_step_1_2','en'] = ('''
               Steps 1 and 2. Search the maximal candidates
               ''')       
    dict_text['p1_alpha_algorithm_step_3','en'] = ('''
               Step 3. Get the set of places $P$
               ''')  
    dict_text['p1_alpha_algorithm_step_4_6','en'] = ('''
               Steps 4 and 6. Get the set of transitions $T$ and labeling function $l$
               ''')      
    dict_text['p1_alpha_algorithm_step_5','en'] = ('''
               Step 5. Get the set of arcs $F$
               ''') 
    dict_text['p1_alpha_algorithm_step_7','en'] = ('''
               Step 7. Get the initial and final marking ($M_{init} = [P I]$, $M_{final} = [P O]$)     
               ''') 
    dict_text['p1_alpha_algorithm_step_8','en'] = ('''
               Step 8. The discovered accepting Petri net $disc_{alpha}(L) = ((P,T,F,l),M_{init},M_{final})$
               ''')               
    dict_text['p1_alpha_algorithm_features_limitations_1','en'] = ('''
               According to the [1]:
               > The complexity of the algorithm is in the first two steps building the sets $Cnd$ and 
               $Sel$ that are used to create the places in Step 3. The Alpha algorithm creates a transition $t_a$
               for each activity $a$ in the event log and also adds a start transition $T0(t_I)$ and an end transition $TN(t_O)$(Step 4). 
               Transitions are labeled with the corresponding activity (Step 6). Transitions $t_I$ and $t_O$ are silent 
               (although we show I and O labels for them, these are dummy labels - there are not the corresponding activities in the event log), 
               $t_I$ has a source place $p_I$ as input and $t_O$ has a sink place $p_O$ as output. The initial marking
               only marks the source place $p_I$ and the final marking only marks the sink place $p_O$ (Step 7). 
               Steps 3–8 can be seen as "bookkeeping". The essence of the algorithm is in the first two steps.    
                    
               To implement steps 1 and 2, we did not go through all the candidates according to step 1 (there may be many) 
               but immediately built a set of the selected maximal places from top to bottom.
               ''')   
    dict_text['p1_alpha_algorithm_features_limitations_2','en'] = ('''
               Technique for constructing a list of pairs $Sel$ (maximal candidates) like $[(\{a\}, \{c\}), (\{c,d\}, \{g\}), ...]$:    
               1. Iterate through all activities in $A' = A \\cup \{I, O\}$.    
               2. For each activity $a_i$, construct the pair $(A_1, A_2)$ as follows:    
                   a. Add the current activity $a_i$ to $A_1$.    
                   b. Select all activities $\{x_1, ..., x_n\}$ connected to $a_i$ as $a_i \\to x_1, ..., a_i \\to x_n$ and add them to $A_2$.    
                   c. Remove self-connected activities $(||)$ from $A_2 = \{x_1, ..., x_n\}$.    
                   d. Select all activities $\{y_1, ..., y_m\}$ connected to $\{x_1, ..., x_n\}$ as $y_j \\to x_k (j = 1...m, k = 1..n)$ and add them to $A_1$.    
                   e. Remove self-connected activities $(||)$ from $A_1 = \{a_i, y_1, ..., y_m\}$.    
                   f. Check that no activities within $A_1$ or $A_2$ are connected to each other $(\#)$. If any are connected, split the set into subsets until # is achieved in each subset 
                      (ultimately, a subset may contain only one activity). For example, $A_1 \\to \{A_1^1, A_1^2\}$ and $A_2 \\to \{A_2^1, A_2^2\}$.     
                   g. From the subsets obtained in step f, create all possible combinations 
                   (e.g., $[(A_1^1, A_2^1), (A_1^1, A_2^2), (A_1^2, A_2^1), (A_1^2, A_2^2)])$. 
                   These are candidate positions (not all possible).      
                   h. For each candidate:      
                      - If the candidate is already in $Sel$, skip it.    
                      - If the candidate is a subset of an existing element in $Sel$ (i.e., the candidate is not maximal), skip it.    
                      - If the candidate is not in $Sel$, check if there are any elements in $Sel$ that are subsets of the candidate. 
                        If so, remove them from $Sel$ and add the candidate to $Sel$.     
                            
                   Note that steps 4 and 6 limit the alpha-algorithm to cycles of length one, while steps 3 and 5 limit it to cycles of length two.
               ''')   
    dict_text['p1_alpha_algorithm_features_limitations_3','en'] = ('''
                   Other limitations of the Alpha-algorithm:    
                   - It does not take frequencies into account (so the algorithm is sensitive to noise and incompleteness).    
                   - It cannot handle skipping (i.e., silent transitions). For example, create a process model for the $L_7$ event log and note the activity $b$.   
                   - It does not discovery non-local dependencies. For example, create a process model for the $L_3$ event log 
                   and explore the model behavior (what about two positions $P_{a\_d}$ and $P_{b\_e}$ not discovered by the Alpha Algorithm).   
                   ''')              
               
    dict_text['p1_summary','en'] = ('''
               **Success! We have accomplished the following:**
                1. Constructed the DFG and its footprint.
                2. Executed the Alpha Algorithm step by step and showed intermediate results as the set of tables.
                3. Visualized the discovered process model as a Petri net graph.                                   
               ''')                
             
    return dict_text

if __name__ == "__main__":
    main()