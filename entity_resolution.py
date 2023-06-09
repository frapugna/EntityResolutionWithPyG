from graph import *
from embeddings import *
from ER import *
def find_matching_couples_free(n_top_A, n_top_B, index_to_token, n_top):
    """
        Output format: ((index_A, index_B, pos_A_to_B, pos_B_to_A), (index_A, index_B))
    """
    matches = []
    matches_no_pos = []
    len_A = len(n_top_A.keys())

    for k in n_top_A.keys():
        A = n_top_A[k]
        for i in range(min(len(A), n_top)):
            B = n_top_B[A[i]]
            pos=-1
            for j in range(min(len(B), n_top)):
                if B[j] == k:
                    pos = j
                    break
            if pos >= 0 and k < A[i]:
                matches.append((index_to_token[k], index_to_token[A[i]]))
                #matches_no_pos.append((index_to_token[k], index_to_token[A[i]]))

    return set(matches)  #, matches_no_pos
def one_to_one_clean_ER(dfpathA, dfpathB,n_epochs=100,p=20, 
                        q=1, n_similar=10, walks_per_node=20,n_top=10, embedding_size=128, walk_length=10, 
                        use_faiss=True,file_directory=None, load_embedding_file=False, load_graph=False, load_n_best=False):
    """
        Params:
        - dfpathA: path to the first dataset 
        - dfpathB: path to the second dataset
        - p: higher values improve exploration during the generation of random walks
        - q: lower values improve exploration during the generation of random walks
        - n_similar: maximum number of closest tuples to find
        - walks_per_node: number of random walks to generate for each node in the graph
        - n_top: number of closest tuples to find (<= n_similar) 
        - embedding_size: the size of the node embeddings
        - walk_length: length of the generated rnadom walks
        - use_faiss: if True tells to use faiss to find the top n closest nodes
        - file_directory: directory where to save the intermediate data
        - load_embedding_file: if true tells to load the embeddings from the file_directory skipping training
        - load_graph: if true tells to load the graph from the file_directory skipping its generation
        - load_n_best: if true tells to load the embeddings from the file_directory skipping their computation
        Output: a set of couples represented the candidate matches found in the format (index, index+len(table_A))
    """
    context_size = walk_length
    # matches = entity_resolution(dfpathA, dfpathB, walks_per_node=walks_per_node, file_directory=None,p=p, q=q, n_epochs=n_epochs, n_similar=n_similar, n_top=n_top, embedding_size=embedding_size, walk_length=walk_length,context_size=context_size, use_faiss=use_faiss, load_embedding_file=load_embedding_file, load_graph=load_graph, load_n_best=load_n_best)
    # return matches
    if not(load_n_best):
        if not(load_embedding_file):
            if not(load_graph):
                dfA = pd.read_csv(dfpathA)
                dfB = pd.read_csv(dfpathB)

                g = Graph()

                g.add_table(dfA,'A', add_table_node=False, link_tuple_token=True, link_token_attribute=True, link_tuple_attribute=False, link_table_tuple=False, link_table_attribute=False,link_table_token=False)
                g.add_table(dfB,'B', add_table_node=False, link_tuple_token=True, link_token_attribute=True, link_tuple_attribute=False, link_table_tuple=False, link_table_attribute=False,link_table_token=False)
                if file_directory != None:
                    g.save(directory_name=file_directory)
            else:
                g = Graph(directory_name=file_directory)
                print('Graph loaded')
        print(f'Number of walks: {g.get_number_of_nodes()}')
        if not(load_embedding_file):
            e = Embeddings(g)
            e.generateNode2vecEmbeddings(n_epochs=n_epochs, p=p, q=q, walks_per_node=walks_per_node, walk_length=walk_length,embedding_dim=embedding_size, context_size=context_size)
            if file_directory != None:
                e.save_embedding_tensor(directory_name=file_directory)
        else:
            e = Embeddings(directory_name=file_directory)
            print('Embeddings loaded')

        tables_to_tuples = get_tuple_indexes_by_table(e.token_to_index)

        if use_faiss:
            print('Using faiss')
            n_top_A = find_top_n_faiss(tables_to_tuples['A'],tables_to_tuples['B'], e.embeddings, n_similar)
            n_top_A = {tables_to_tuples['A'][i]:[tables_to_tuples['B'][t] for t in n_top_A[i]] for i in range(n_top_A.shape[0])}

            n_top_B = find_top_n_faiss(tables_to_tuples['B'],tables_to_tuples['A'], e.embeddings, n_similar)
            n_top_B = {tables_to_tuples['B'][i]:[tables_to_tuples['A'][t] for t in n_top_B[i]] for i in range(n_top_B.shape[0])}
        else:
            n_top_A = find_n_best(e, tables_to_tuples['A'], tables_to_tuples['B'], n_similar)
            n_top_B = find_n_best(e, tables_to_tuples['B'], tables_to_tuples['A'], n_similar)
        index_to_token = e.index_to_token
        if file_directory != None:
            save_n_top(file_directory, n_top_A, n_top_B, e.index_to_token)
    if file_directory != None:
        n_top_A, n_top_B, index_to_token = load_n_top(file_directory)
    
    matches = find_matching_couples_free(n_top_A, n_top_B, index_to_token, n_top)
    if file_directory != None:
        save_matches(matches, file_directory)
        
    return matches


def free_entity_resolution(df_list,file_directory=None,n_epochs=100,p=20, q=1, 
                        n_top=10, embedding_size=128, walk_length=10, 
                           load_embedding_file=False, load_graph=False):
    """
        Params:
        - df_list: list of the path to the dataframe to process 
        - file_directory: directory where to save the intermediate data
        - n_epochs: the number of training epochs
        - p: higher values improve exploration during the generation of random walks
        - q: lower values improve exploration during the generation of random walks
        - n_top: number of closest tuples to find (<= n_similar) 
        - embedding_size: the size of the node embeddings
        - walk_length: length of the generated rnadom walks
        - load_embedding_file: if true tells to load the embeddings from the file_directory skipping training
        - load_graph: if true tells to load the graph from the file_directory skipping its generation
        Output: a set of couples represented the candidate matches found in the format (tp_<table_from>_<index>,tp_<table_from>_<index>)
    """
    context_size=walk_length
    n_similar = n_top
    if not(load_embedding_file):
        if not(load_graph):
            df_list = [pd.read_csv(df_list[i]) for i in range(len(df_list))]
            g = Graph()
            for i in range(len(df_list)):
                g.add_table(df_list[i],str(i), add_table_node=False, link_tuple_token=True, link_token_attribute=True, link_tuple_attribute=False, link_table_tuple=False, link_table_attribute=False,link_table_token=False)

            if file_directory != None:
                g.save(directory_name=file_directory)
        else:
            g = Graph(directory_name=file_directory)
            print('Graph loaded')
    print(f'Number of walks: {g.get_number_of_nodes()}')
    if not(load_embedding_file):
        e = Embeddings(g)
        e.generateNode2vecEmbeddings(n_epochs=n_epochs, p=p, q=q,walk_length=walk_length,embedding_dim=embedding_size, context_size=context_size)
        if file_directory != None:
            e.save_embedding_tensor(directory_name=file_directory)
    else:
        e = Embeddings(directory_name=file_directory)
        print('Embeddings loaded')

    tables_to_tuples = get_tuple_indexes_by_table(e.token_to_index)
    index_list = []
    for k,v in tables_to_tuples.items():
        index_list+=v
    n_top_pairs = find_top_n_faiss(index_list, index_list, e.embeddings, n_similar)

    n_top_pairs = {index_list[i]:[index_list[t] for t in n_top_pairs[i] ] for i in range(n_top_pairs.shape[0])}

    index_to_token = e.index_to_token

    
    matches = find_matching_couples_free(n_top_pairs,n_top_pairs, index_to_token, n_top)

    if file_directory != None:
        save_matches(matches, file_directory)
    try:
        tot_comp = 1
        for i in range(len(df_list)):
            tot_comp*=df_list[i].shape[0]
        print(f'Old number of candidates: {tot_comp}\nNew number of candidates: {len(matches)}')
    except:
        print('Reduction ratio computation failed')
    return matches

if __name__ == '__main__':
    #matches = one_to_one_clean_ER(r"C:\Users\frapu\Desktop\ProgettoBeneventano\Tests\FZ\Datasets\fodors_zagats-tableA.csv", r"C:\Users\frapu\Desktop\ProgettoBeneventano\Tests\FZ\Datasets\fodors_zagats-tableB.csv")
    #matches = free_entity_resolution([r"C:\Users\frapu\Desktop\ProgettoBeneventano\Tests\FZ\Datasets\fodors_zagats-tableA.csv", r"C:\Users\frapu\Desktop\ProgettoBeneventano\Tests\FZ\Datasets\fodors_zagats-tableB.csv"], n_epochs=1)
    #print(matches)
    matches = free_entity_resolution([r"C:\Users\frapu\Desktop\ProgettoBeneventano\Datasets\testAB.csv", r"C:\Users\frapu\Desktop\ProgettoBeneventano\Datasets\testBC.csv"], n_epochs=1)
    print(matches)
    print('end')