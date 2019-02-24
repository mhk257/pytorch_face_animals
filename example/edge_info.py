import numpy as np

def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2],dtype=np.float32)
    #print(edges)
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] =  1
    return a


def gen_links(edges, ii):

    edges.append([ii, 1, ii+1])
    edges.append([ii+1, 1, ii])
    edges.append([ii, 2, ii+1])
    edges.append([ii+1, 2, ii])

    return edges

def gen_links2(edges, ii, jj):

    edges.append([ii, 1, jj])
    edges.append([jj, 1, ii])
    edges.append([ii, 2, jj])
    edges.append([jj, 2, ii])

    return edges






def create_edges(pairwise, symmetrical):


    edges = []

    if pairwise:
        ######### Pairwise connections ###############

        print('Pairwise connections..........')

        for ii in range(1,17):

            gen_links(edges, ii)

        for ii in range(18,22):

            gen_links(edges, ii)

        for ii in range(23,27):

            gen_links(edges, ii)

        for ii in range(37,40):

            gen_links(edges, ii)    

        
        gen_links2(edges, 37, 42)
        gen_links2(edges, 42, 41)
        gen_links2(edges, 41, 40)
        

        for ii in range(43,46):

            gen_links(edges, ii)

        gen_links2(edges, 43, 48)
        gen_links2(edges, 48, 47)
        gen_links2(edges, 47, 46)
        
        for ii in range(28,31):

            gen_links(edges, ii)

        for ii in range(32,36):

            gen_links(edges, ii)

        for ii in range(49,55):

            gen_links(edges, ii)

        gen_links2(edges, 49, 61)
        gen_links2(edges, 61, 62)
        gen_links2(edges, 62, 63)
        gen_links2(edges, 63, 64)
        gen_links2(edges, 64, 65)
        gen_links2(edges, 65, 55)
        
        gen_links2(edges, 49, 68)
        gen_links2(edges, 68, 67)
        gen_links2(edges, 67, 66)
        gen_links2(edges, 66, 55)
        
        
        gen_links2(edges, 49, 60)
        gen_links2(edges, 60, 59)
        gen_links2(edges, 59, 58)
        gen_links2(edges, 58, 57)
        gen_links2(edges, 57, 56)
        gen_links2(edges, 56, 55)
        

    if symmetrical:


        print('Symmetry connections..........')

        gen_links2(edges, 18, 27)
        gen_links2(edges, 19, 26)
        gen_links2(edges, 20, 25)
        gen_links2(edges, 21, 24)
        gen_links2(edges, 22, 23)

        gen_links2(edges, 37, 46)
        gen_links2(edges, 39, 45)
        gen_links2(edges, 41, 47)
        gen_links2(edges, 42, 48)
        
        gen_links2(edges, 32, 36)
        gen_links2(edges, 33, 35)
        


        gen_links2(edges, 49, 55)
        gen_links2(edges, 61, 65)
        gen_links2(edges, 50, 54)

        gen_links2(edges, 51, 53)
        gen_links2(edges, 62, 64)
        gen_links2(edges, 68, 66)

        gen_links2(edges, 60, 56)
        gen_links2(edges, 59, 57)

        # symmetrical edges (1 -> 17)
        edges.append([1, 1 ,17])
        edges.append([17, 1 ,1])
        edges.append([1, 2 ,17])
        edges.append([17, 2 ,1])

        edges.append([2, 1 ,16])
        edges.append([16, 1 ,2])
        edges.append([2, 2 ,16])
        edges.append([16, 1 ,2])

        edges.append([3, 1 ,15])
        edges.append([15, 1 ,3])
        edges.append([3, 2 ,15])
        edges.append([15, 2 ,3])

        edges.append([4, 1 ,14])
        edges.append([14, 1 ,1])
        edges.append([4, 2 ,14])
        edges.append([14, 2 ,1])

        edges.append([5, 1 ,13])
        edges.append([13, 1 ,5])
        edges.append([5, 2 ,13])
        edges.append([13, 2 ,5])

        edges.append([6, 1 ,12])
        edges.append([12, 1 ,6])
        edges.append([6, 2 ,12])
        edges.append([12, 2 ,6])

        edges.append([7, 1 ,11])
        edges.append([11, 1 ,7])
        edges.append([7, 2 ,11])
        edges.append([11, 2 ,7])

        edges.append([8, 1 ,10])
        edges.append([10, 1 ,8])
        edges.append([8, 2, 10])
        edges.append([10, 2 ,8])

    print('Total # of edges --> {}'.format(len(edges)))
       
 
    #print(len(edges))


    

    return edges




edges1 = create_edges(pairwise=True, symmetrical=False)


adj_matrix = create_adjacency_matrix(edges1, 68, 2)


for kk in range(68):
    print(adj_matrix[kk,:68])

