import pandas as pd
from graph import *
from visualization import *
from embeddings import *

#df = pd.read_csv(r"C:\Users\frapu\Desktop\GeneralPurposeTableEmbedding\toyDataset.csv")
#dfA = pd.read_csv(r"C:\Users\frapu\Desktop\GeneralPurposeTableEmbedding\Datasets\testAB.csv")
#dfB = pd.read_csv(r"C:\Users\frapu\Desktop\GeneralPurposeTableEmbedding\Datasets\testBC.csv")
print('Start')
dfA = pd.read_csv(r"/home/francesco.pugnaloni/GeneralPurposeTableEmbedding/Datasets/testAB.csv")
#dfB = pd.read_csv(r"C:\Users\frapu\Desktop\GeneralPurposeTableEmbedding\Datasets\fodors_zagats-tableB.csv")

g = Graph()
#df = df.iloc[[0]]

#print(df.head())

#g.add_table(df.iloc[[0]], 'my_table',add_table_node=True, link_tuple_token=False, link_token_attribute=False, link_tuple_attribute=False, link_table_tuple=False, link_table_attribute=False, link_table_token=False)
#g.add_table(df.iloc[[0]], 'my_table')

g.add_table(dfA, 'A')
#g.add_table(dfB, 'B')
print('Table added')
#print(str(g))
#print(g.token_to_index)

visualize_graph(g)

#e = Embeddings(g)
#e.generateNode2vecEmbeddings(n_epochs=10)

#print('end')