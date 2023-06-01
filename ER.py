from graph import *
from embeddings import *
from visualization import *
from research_similar import *
import sys

def get_tuple_indexes_by_table(token_to_index, add_also_indexes=False):
    out = {}
    for k in token_to_index.keys():
        split_str = k.split('_')
        if split_str[0] == 'tp':
            try:
                out[split_str[1]].append(token_to_index[k])
            except:
                out[split_str[1]]=[]
                out[split_str[1]].append(token_to_index[k])
    return out

def find_n_best(embeddings, list_from, list_to, n_top):
    n_top_A = {}
    n_iter = 0  #Debug
    n_elements = len(list_from) #Debug
    for t_A in list_from:
        n_top_A[t_A] = []
        if n_iter%1==0: #Debug
            print(f'Iter number: {n_iter} / {n_elements}') #Debug
        for t_B in list_to:
            similarity = compare_embeddings(embeddings[t_A], embeddings[t_B])
            i = len(n_top_A[t_A]) - 1
            while i >= 0:
                if n_top_A[t_A][i][1] >= similarity:
                    break
                i-=1

            if i < n_top-1:
                n_top_A[t_A].insert(i+1, (t_B,similarity))

                try:
                    n_top_A[t_A].pop(n_top)
                except:
                    pass
        n_iter += 1 #Debug
    return n_top_A

def save_n_top(directory_name, n_top_A, n_top_B, index_to_token):
    try:
        f1 = open(directory_name+'/n_top_A.pkl', 'wb')
        f2 = open(directory_name+'/n_top_B.pkl', 'wb')
        f3 = open(directory_name+'/index_to_token.pkl', 'wb')
        
        pickle.dump(n_top_A, f1)
        pickle.dump(n_top_B, f2)
        pickle.dump(index_to_token, f3)

        f1.close()
        f2.close()
        f3.close()
    except:
        raise Exception('Embedding write operation failed')

def load_n_top(directory_name):
    try:
        f1 = open(directory_name+'/n_top_A.pkl', 'rb')
        f2 = open(directory_name+'/n_top_B.pkl', 'rb')
        f3 = open(directory_name+'/index_to_token.pkl', 'rb')

        n_top_A = pickle.load(f1)
        n_top_B = pickle.load(f2)
        index_to_token = pickle.load(f3)
        
        f1.close()
        f2.close()
        f3.close()
    except:
        raise Exception('Embedding read operation failed')
    return n_top_A, n_top_B, index_to_token

def find_matching_couples(n_top_A, n_top_B, index_to_token, n_top, faiss=True, embdi_format=True):
    """
        Output format: ((index_A, index_B, pos_A_to_B, pos_B_to_A), (index_A, index_B))
    """
    matches = []
    matches_no_pos = []
    len_A = len(n_top_A.keys())
    if not(faiss):
        for k in n_top_A.keys():
            A = n_top_A[k]
            for i in range(min(len(A), n_top)):
                B = n_top_B[A[i][0]]
                pos=-1
                for j in range(min(len(B), n_top)):
                    if B[j][0] == k:
                        pos = j
                        break
                if pos >= 0:
                    matches.append((index_to_token[k].split('_')[2], index_to_token[A[i][0]].split('_')[2], i, pos))
                    matches_no_pos.append((int(index_to_token[k].split('_')[2]), int(index_to_token[A[i][0]].split('_')[2])+len_A))
    else:
        for k in n_top_A.keys():
            A = n_top_A[k]
            for i in range(min(len(A), n_top)):
                B = n_top_B[A[i]]
                pos=-1
                for j in range(min(len(B), n_top)):
                    if B[j] == k:
                        pos = j
                        break
                if pos >= 0:
                    if embdi_format:
                        matches.append((index_to_token[k].split('_')[2], index_to_token[A[i]].split('_')[2], i, pos))
                        matches_no_pos.append((int(index_to_token[k].split('_')[2]), int(index_to_token[A[i]].split('_')[2])+len_A))
                    else:
                        matches.append((index_to_token[k], index_to_token[A[i]]))
                        matches_no_pos.append((index_to_token[k], index_to_token[A[i]]))

    return matches, matches_no_pos

def save_matches(matches, directory_name):
    try:
        f1 = open(directory_name+'/matches.pkl', 'wb')
        pickle.dump(matches, f1)
        f1.close()
    except:
        raise Exception('Matches write operation failed')
    
def load_matches(directory_name):
    try:
        f1 = open(directory_name+'/matches.pkl', 'rb')
        out = pickle.load(f1)
        f1.close()
    except:
        raise Exception('Matches write operation failed')
    return out


def prepare_ground_truth_embdi(file_path):
    import re
    ground_truth = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove newline character
            # Extract two numbers using regular expressions
            matches = re.findall(r'\d+', line)
            if len(matches) >= 2:
                ground_truth.append((int(matches[0]), int(matches[1])))
    return set(ground_truth)


def entity_resolution(dfpathA, dfpathB, p=20, q=1, n_epochs=100, n_similar=10, n_top=10, embedding_size=128, walk_length=10,context_size=10, walks_per_node=20,return_n_pairs_before=False, use_faiss=True,file_directory=None, load_embedding_file=False, load_graph=False, load_n_best=False):
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
    
    matches = find_matching_couples(n_top_A, n_top_B, index_to_token, n_top,use_faiss)
    if file_directory != None:
        save_matches(matches, file_directory)
    
    if return_n_pairs_before:    
        try:
            return set(matches[1]), dfA.shape[0]*dfB.shape[0]
        except:
            dfA = pd.read_csv(dfpathA)
            dfB = pd.read_csv(dfpathB)
            return set(matches[1]), dfA.shape[0]*dfB.shape[0]
        
    return set(matches[1])

def measure_performances(matches, ground_truth, number_of_pairs_before):
    found = 0
    for t in ground_truth:
        t_alt = (t[1], t[0])    #note: to be compliant to embdi labeling you have to subtract to t[1] the number of elements of the first table
        if t in matches or t_alt in matches:
            found += 1

    precision = found / len(matches)
    recall = found / len(ground_truth) 
    try:
        f_measure = (2*(precision*recall))/(precision+recall)
    except:
        f_measure = -1
        print('Pairs quality or pairs completeness is 0')
    reduction_ratio= 1-(len(matches)/number_of_pairs_before)
  
    print(f'Number of pairs pre blocking: {number_of_pairs_before}')
    print(f'Number of pairs post blocking: {len(matches)}')
    print(f'Pairs completeness: {recall}')
    print(f'Reduction ratio: {reduction_ratio}')

    print('_________________________________________________________________________')
    #print(f'Precision: {precision} Recall: {recall} F-measure: {f_measure}')

def find_top_n_faiss(table_to_tuples_A, table_to_tuples_B, embeddings, n_similar):
    embeddings = embeddings.detach().cpu().numpy()
    faiss_embeddings_A = embeddings[table_to_tuples_A]

    faiss_embeddings_B = embeddings[table_to_tuples_B]

    distances, corpus_ids = find_n_top_A_B_exact(faiss_embeddings_A, faiss_embeddings_B, n_similar)

    return corpus_ids




def run_test(dfpathA, dfpathB, ground_truth_path, file_directory, p=20,q=1, embedding_size=128, walk_length=10,n_similar=10, n_top=10, n_epochs=100, load_embedding_file=False, load_graph=False, load_n_best=False):
    matches, n_pairs_before = entity_resolution(dfpathA, dfpathB, return_n_pairs_before=True,p=p, q=q, embedding_size=embedding_size, walk_length=walk_length, file_directory=file_directory, n_epochs=n_epochs,n_similar=n_similar, n_top=n_top, load_graph=load_graph, load_embedding_file=load_embedding_file, load_n_best=load_n_best)
    ground_truth = prepare_ground_truth_embdi(ground_truth_path)
    measure_performances(matches, ground_truth, n_pairs_before)



if __name__ == '__main__':
    try:
        task = sys.argv[1]
    except:
        task = 'FZ-test'

    if task == 'FZ-train-test':
        start = time.time()
        run_test(r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Datasets/fodors_zagats-tableA.csv",
            r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Datasets/fodors_zagats-tableB.csv",
            r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Datasets/matches-fodors_zagats.txt",
            r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Files",
            n_epochs=100,
            load_n_best=False,
            load_embedding_file=False,
            load_graph=False,
            n_similar=10,
            n_top=10,
            embedding_size=300,
            walk_length=10,
            p=10,
            q=1
            )
        
        end = time.time()
        print(f'Texec: {end-start}')
    if task == 'FZ-test':
        for i in range(1,11):
            print(f'n_top: {i}')
            run_test(r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Datasets/fodors_zagats-tableA.csv",
                r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Datasets/fodors_zagats-tableB.csv",
                r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Datasets/matches-fodors_zagats.txt",
                r"/home/francesco.pugnaloni/EntityResolutionWithPyG/Tests/FZ/Files",
                n_epochs=100,
                load_n_best=True,
                load_embedding_file=False,
                load_graph=False,
                n_similar=10,
                n_top=i,
                embedding_size=300,
                walk_length=10,
                p=10,
                q=1
                )
    if task == 'beer-train-test':
        start = time.time()
        run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Datasets/beer-tableA.csv",
            r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Datasets/beer-tableB.csv",
            r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Datasets/matches-beer.txt",
            r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Files",
            n_epochs=70,
            load_n_best=False,
            load_embedding_file=False,
            load_graph=False,
            n_similar=10,
            n_top=10,
            embedding_size=300,
            walk_length=10,
            p=10,
            q=1
            )
        end = time.time()
        print(f'Texec: {end-start}')

    if task == 'beer-test':
        for i in range(1,11):
            print(f'n_top: {i}')
            run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Datasets/beer-tableA.csv",
                r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Datasets/beer-tableB.csv",
                r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Datasets/matches-beer.txt",
                r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/Beer/Files",
                n_epochs=100,
                load_n_best=True,
                load_embedding_file=False,
                load_graph=False,
                n_similar=10,
                n_top=i,
                embedding_size=128,
                walk_length=10,
                )
    if task == 'AG-test':
        for i in range(1,11):
            print(f'n_top: {i}')
            run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Datasets/amazon_google-tableA.csv",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Datasets/amazon_google-tableB.csv",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Datasets/matches-amazon_google.txt",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Files",
                        n_epochs=70,
                        load_n_best=True,
                        n_similar=10,
                        n_top=i,
                        embedding_size=300,
                        walk_length=10,
                        )
    if task == 'AG-train-test':
        print('Going to run AZ-train-test')
        run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Datasets/amazon_google-tableA.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Datasets/amazon_google-tableB.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Datasets/matches-amazon_google.txt",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/AG/Files",
                    n_epochs=70,
                    load_n_best=False,
                    load_embedding_file=False,
                    load_graph=False,
                    n_similar=10,
                    n_top=10,
                    embedding_size=300,
                    walk_length=10,
                    )
        
    if task == 'dblp_acm-test':
        for i in range(1,11):
            print(f'n_top: {i}')
            run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Datasets/dblp_acm-tableA.csv",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Datasets/dblp_acm-tableB.csv",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Datasets/matches-dblp_acm.txt",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Files",
                        n_epochs=70,
                        load_n_best=True,
                        n_similar=10,
                        n_top=i,
                        embedding_size=300,
                        walk_length=10,
                        )
            
    if task == 'dblp_acm-train-test':
        run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Datasets/dblp_acm-tableA.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Datasets/dblp_acm-tableB.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Datasets/matches-dblp_acm.txt",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/dblp_acm/Files",
                    n_epochs=200,
                    load_n_best=False,
                    load_embedding_file=False,
                    load_graph=False,
                    n_similar=10,
                    n_top=10,
                    embedding_size=300,
                    walk_length=10,
                    )
    if task == 'WA-test':
        for i in range(1,11):
            print(f'n_top: {i}')
            run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Datasets/walmart_amazon-tableA.csv",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Datasets/walmart_amazon-tableB.csv",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Datasets/matches-walmart_amazon.txt",
                        r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Files",
                        n_epochs=70,
                        load_n_best=True,
                        n_similar=10,
                        n_top=i,
                        embedding_size=300,
                        walk_length=10,
                        )
            
    if task == 'WA-train-test':
        run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Datasets/walmart_amazon-tableA.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Datasets/walmart_amazon-tableB.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Datasets/matches-walmart_amazon.txt",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Tests/WA/Files",
                    n_epochs=70,
                    load_n_best=False,
                    load_embedding_file=False,
                    load_graph=False,
                    n_similar=10,
                    n_top=10,
                    embedding_size=300,
                    walk_length=10,
                    )
    if task == 'toy-dataset':
        run_test(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Datasets/testAB.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Datasets/testBC.csv",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Datasets/prova.txt",
                    r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/TestDir",
                    n_epochs=1,
                    load_n_best=False,
                    n_similar=10,
                    n_top=10,
                    embedding_size=128,
                    walk_length=10,
                    )
