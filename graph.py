import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import math
from bert_embeddings import *

"""
    Assumptions:
    - the entire graph can fit in memory
    - there are not 2 tables with the same name and if 
    - all the columns of the dataframe have a string as identifier
    - all the columns inside the same dataframe have different names
    - the graph is undirected and unweighted
"""
def isNaN(num):
    return num != num

def get_order_of_magnitude(number):
    if number == 0:
        return 0  # Logarithm of 0 is undefined, return 0 as the order of magnitude
    else:
        return int(math.floor(math.log10(abs(number))))

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def preprocess_numbers(n, operations=['cast_to_float', 'discretize_strict']):
    if 'cast_to_float' in operations:
        n = float(n)
    if 'discretize_strict' in operations:
        div = 10 ** get_order_of_magnitude(n)
        n = n//div*div
    return n

class String_token_preprocessor:
    def __init__(self, language='english'):
        nltk.download('stopwords')
        self.stopwords = stopwords.words(language)

    def preprocess_attribute(self, s, operations=['lowercase', 'drop_numbers_from_strings']):
        out = str(s)
        if 'lowercase' in  operations:
            out = out.lower()
        if 'drop_numbers_from_strings' in operations:
            if not(is_float(out)):
                out = self.__call__(out)
                out = '_'.join([t for t in out if not(is_float(t))])
        return out

    def __call__(self, s, operations=['lowercase', 'split', 'remove_stop_words']):
        out = s
        if len(operations) == 0:
            return [out]
        
        if 'lowercase' in operations:
            out = out.lower()

        if 'split' in operations:
            out = re.split(' |_|\|', out)
        
        if 'remove_stop_words' in operations:
            out = [t for t in out if not(t in self.stopwords)]

        return out

class Graph:
    """
        Node prefixes:
        - tb_: tables
        - tp_<table_from>_: tuples
        - at_: attributes
        - tk_: tokens
    """

    def __init__(self, directory_name=False, task='entity-resolution'):
        """
            Params:
            -directory_name: path to a directory containing all the file necessary to istantiate a prebuilt graph, if False an empty graph will be istantiated
        """
        if directory_name != False:
            self.load(directory_name)
        else:
            self.edges = [[],[]]
            self.number_of_tokens = 0
            self.number_of_edges = 0
            self.index_to_token = {}
            self.token_to_index = {}
            self.preprocess_string_token = String_token_preprocessor()
            self.task = task
            if self.task == 'table-matching':
                self.generate_embedding = Bert_Embedding_Generator()
    def get_number_of_nodes(self):
        return len(self.index_to_token)
    def __str__(self):
        return ''.join(f'{self.index_to_token[self.edges[0][i]]}<-->{self.index_to_token[self.edges[1][i]]}\n' for i in range(self.number_of_edges))
    
    def add_token(self, token):
        try:
            index = self.token_to_index[token]
        except:
            self.token_to_index[token] = self.number_of_tokens
            self.index_to_token[self.number_of_tokens] = token
            index = self.number_of_tokens
            self.number_of_tokens += 1
        return index
    
    def add_edge(self, id_a, id_b):
        self.edges[0].append(id_a)
        self.edges[1].append(id_b)
        self.number_of_edges += 1
            
        self.edges[0].append(id_b)
        self.edges[1].append(id_a)
        self.number_of_edges += 1
    
    def add_table(self, df, table_name='', config='embdi',add_table_node=True, link_tuple_token=True, link_token_attribute=True, link_tuple_attribute=False, link_table_tuple=True, link_table_attribute=True
                  , link_table_token=False, attribute_preprocess_operations = ['lowercase', 'drop_numbers_from_strings'], string_preprocess_operations = ['lowercase', 'split', 'remove_stop_words'],
                  number_preprocess_operations = ['cast_to_float', 'discretize_strict']):
        if self.task == 'table-matching':
            self.__add_table_TM(df, table_name=table_name, config=config,add_table_node=add_table_node, link_tuple_token=link_tuple_token, link_token_attribute=link_token_attribute, link_tuple_attribute=link_tuple_attribute, link_table_tuple=link_table_tuple, link_table_attribute=link_table_attribute
                  , link_table_token=link_table_token, attribute_preprocess_operations = attribute_preprocess_operations, string_preprocess_operations = string_preprocess_operations,
                  number_preprocess_operations = number_preprocess_operations)

        elif self.task == 'entity-resolution':
            self.__add_table_ER(df, table_name=table_name, config=config,add_table_node=add_table_node, link_tuple_token=link_tuple_token, link_token_attribute=link_token_attribute, link_tuple_attribute=link_tuple_attribute, link_table_tuple=link_table_tuple, link_table_attribute=link_table_attribute
                  , link_table_token=link_table_token, attribute_preprocess_operations = attribute_preprocess_operations, string_preprocess_operations = string_preprocess_operations,
                  number_preprocess_operations = number_preprocess_operations)
            
    def __add_table_TM(self, df, table_name='', config='embdi',add_table_node=True, link_tuple_token=True, link_token_attribute=True, link_tuple_attribute=False, link_table_tuple=True, link_table_attribute=True
                  , link_table_token=False, attribute_preprocess_operations = ['lowercase', 'drop_numbers_from_strings'], string_preprocess_operations = ['lowercase', 'split', 'remove_stop_words'],
                  number_preprocess_operations = ['cast_to_float', 'discretize_strict']):
        return
        
    def __add_table_ER(self, df, table_name='', config='embdi',add_table_node=True, link_tuple_token=True, link_token_attribute=True, link_tuple_attribute=False, link_table_tuple=True, link_table_attribute=True
                  , link_table_token=False, attribute_preprocess_operations = ['lowercase', 'drop_numbers_from_strings'], string_preprocess_operations = ['lowercase', 'split', 'remove_stop_words'],
                  number_preprocess_operations = ['cast_to_float', 'discretize_strict']):
        """
            Desc: a dataframe will be processed to generate nodes and edges to add to the graph
            Params:
            -df: the dataframe to process
            -table_name: the name of the dataframe, it will be used during the node generation
            -add_table_node: if true tells the function to add a node representing the table
            -link_tuple_token: if true tuples and tokens will be linked by edges
            -link_token_attribute: if true tokens and attributes will be linked by edges
            -link_tuple_attribute: if true tuples and attributes will be linked by edges
            -link_table_tuple: if true tables and tuples will be linked by edges
            -link_table_attribute: if true tables and attributes will be linked by edges
            -link_table_token: if true table and tokens will be linked by edges
        """
        if config == 'embdi':
            add_table_node = (self.task=='table_matching')
            link_tuple_token = True
            link_token_attribute = True
            link_tuple_attribute = False
            link_table_token = False
            link_table_attribute = False
            link_table_attribute = False
            attribute_preprocess_operations = ['lowercase', 'drop_numbers_from_strings']
            string_preprocess_operations = ['lowercase', 'split', 'remove_stop_words']
            number_preprocess_operations = ['cast_to_float', 'discretize_strict']

        #Table nodes generation
        table_index = -1
        if add_table_node==True: 
            if table_name=='':
                raise Exception("You need to provide a name for the table if you want to add a table node")
            table_token = 'tb_' + table_name
            table_index = self.add_token(table_token)

        #Attribute nodes generation
        column_indexes = []
        for c in df.columns:
            attribute_name = self.preprocess_string_token.preprocess_attribute(c, operations=attribute_preprocess_operations)
            attribute_token = 'at_' + attribute_name
            attribute_index = self.add_token(attribute_token)
            column_indexes.append(attribute_index)
            if link_table_attribute:
                self.add_edge(table_index, attribute_index)

        #Tuple and token node
        for i in range(df.shape[0]):
            tuple_token = 'tp_' + table_name + '_' + str(i)
            tuple_index = self.add_token(tuple_token)
            
            if add_table_node and link_table_tuple:
                self.add_edge(tuple_index, table_index)

            if link_tuple_attribute:
                for id in column_indexes:
                    self.add_edge(tuple_index, id)
            
            for j in range(df.shape[1]):
                t = df.iloc[i][j]
                if pd.isnull(t):
                    continue

                if isinstance(t, str) and not(is_float(t)):
                    token_list = self.preprocess_string_token(t, operations=string_preprocess_operations)

                elif is_float(str(t)):
                    #Note: the string "infinity" will trigger an exception and will be skipped
                    try:
                        token_list = [preprocess_numbers(t, operations=number_preprocess_operations)]
                    except:
                        print(f'An exception occurred in the position [{i},{j}] of the table {table_name} ')
                        continue

                else:
                    raise Exception(f'The token {t} is of type {type(t)} and it is not supported')
                
                for slice in token_list:
                    token_token = 'tk_' + str(slice)
                    token_index = self.add_token(token_token)

                    if link_tuple_token:
                        self.add_edge(token_index, tuple_index)
                    if link_table_token:
                        self.add_edge(token_index, table_index)
                    if link_token_attribute:
                        self.add_edge(token_index, column_indexes[j])
    
    def save(self, directory_name):
        """
            Desc: the graph will be saved in the provided directory
            Params:
            -directory_name: the path to the directory where to save the graph
        """
        try:
            f1 = open(directory_name+'/token_to_index.pkl', 'wb')
            f2 = open(directory_name+'/index_to_token.pkl', 'wb')
            f3 = open(directory_name+'/edges.pkl', 'wb')
            pickle.dump(self.token_to_index, f1)
            pickle.dump(self.index_to_token, f2)
            pickle.dump(self.edges, f3)
            f1.close()
            f2.close()
            f3.close()
        except:
            raise Exception("Graph write operation failed")
    
    def load(self, directory_name):
        """
            Desc: the graph will be loaded from the provided directory
            Params:
            -directory_name: the path to the directory where the graph is saved
        """
        try:
            f1 = open(directory_name+'/token_to_index.pkl', 'rb')
            f2 = open(directory_name+'/index_to_token.pkl', 'rb')
            f3 = open(directory_name+'/edges.pkl', 'rb')
            self.token_to_index = pickle.load(f1)
            self.index_to_token = pickle.load(f2)
            self.number_of_tokens = len(self.token_to_index)
            self.edges = pickle.load(f3)
            self.number_of_edges = len(self.edges[0])

            f1.close()
            f2.close()
            f3.close()
        except:
            raise Exception("Graph read operation failed")

    
if __name__ == '__main__':
    try:
        df = pd.read_csv(r'/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Datasets/testAB.csv')
    except:
        df = pd.read_csv(r'C:\Users\frapu\Desktop\GeneralPurposeTableEmbedding\Datasets\testAB.csv')
    g = Graph()
    g.add_table(df, 'A', config='collaborer')
    from visualization import *
    visualize_graph(g)
    print('Finish')