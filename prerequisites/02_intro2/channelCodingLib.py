import numpy as np
import numpy as np

# Exercise 1
def cyclgenmat(gp: np.array, m: int, systematic=False) -> np.ndarray:
    """Takes 2 input parameters: gp is a vector of length m-n+1 containing generator polynomial coefficients,
    and m is the code word length. Returns a generator matrix (numpy matrix).
    If the input parameters are in an incorrect format, returns an empty matrix."""
    # Calculate n
    n = m-gp.size+1
    #print(n)
    gen_matrix = np.zeros((n,m),dtype=int)
    for i,val in enumerate(gen_matrix):
        gen_matrix[i][i:(i+(m-n+1))] = gp

    #Construct the parity-check matrix
    sysmat = genmatsys(gen_matrix)

    PTrans = sysmat[:,n:].T
    idmat = np.identity(m-n,dtype=int)
    H = np.concatenate((PTrans,idmat),axis=1)
    
    if not systematic: return gen_matrix #parity-check matrix is only valid if calculated from Gaussian normal form
    else: return (sysmat,H)


# Exercise 2
def genmatsys(G: np.ndarray) ->np.ndarray:
    """Gaussian-Jordan algorithm for getting the reduced row echelon form
     (Gaussian normal form) of the input matrix."""
    #make copy of the matrix for returning
    sys_gen_matrix = np.copy(G)
    
    #get the number of rows and columns of the matrix
    num_rows, num_cols = sys_gen_matrix.shape
    
    #initialize a pointer for the current pivot row
    pivot_row = 0
    
    #iterate over each column to form the identity matrix on the left
    for col in range(num_cols):
        #find the pivot element in the current column
        pivot_index = pivot_row
        while pivot_index < num_rows and sys_gen_matrix[pivot_index][col] == 0:
            pivot_index += 1
        
        #if a pivot element is found, perform row operations
        if pivot_index < num_rows:
            #swap rows to move the pivot element to the current pivot row
            if pivot_index != pivot_row:
                sys_gen_matrix[[pivot_row, pivot_index]] = sys_gen_matrix[[pivot_index, pivot_row]]
            
            #make the pivot element 1 by dividing the row by its value
            #pivot_value = sys_gen_matrix[pivot_row][col]
            #sys_gen_matrix[pivot_row] = sys_gen_matrix[pivot_row] / pivot_value
            
            #eliminate other non-zero elements in the current column
            for i in range(num_rows):
                if i != pivot_row and sys_gen_matrix[i][col] != 0:
                    #sys_gen_matrix[i] -= sys_gen_matrix[pivot_row] * sys_gen_matrix[i][col]
                    sys_gen_matrix[i] = sys_gen_matrix[i] ^ sys_gen_matrix[pivot_row]
            
            #move to the next pivot row
            pivot_row += 1
    
    return sys_gen_matrix


# Exercise 4
def encoder(G: np.array, x: np.array):
    return np.dot(x,G)%2


def de2bi(decimal, n=None):
    """Convert decimal numbers to binary vectors

        Arguments:
            decimal -- integer or iterable of integers
            [n] -- optional, maximum length of representation

        Returns:
            r x n ndarray where each row is a binary vector

        Example:
            de2bi(5)
            -> array([1, 0, 1])
            de2bi(range(6))
            array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1]])
    """

    if type(decimal) is int:
        decimal = (decimal,)

    if n is None:
        # calculate max length of binary representation
        n = int(np.ceil(np.log2(np.max(decimal)+1)))

    # create output matrix
    x = np.zeros(shape=(len(decimal), n), dtype=int)
    for i in range(len(decimal)):
        b = bin(decimal[i])[2:]
        x[i, (n-len(b)):] = np.array(list(b))

    return x


# Exercise 5
def codefeatures(G:np.array):
    m = G.shape[0]
    n = G.shape[1]

    infowordarray = np.array(de2bi(i,m) for i in range(2**m))

    encoded = encoder(infowordarray,G)

    weights = np.sum(encoded,axis=1)

    min_dist = np.min(weights[weights>0])

    #amount of correctable bit errors
    K = (min_dist -1) // 2

    return min_dist,K


# Exercise 6
def syndict(H, K):
    #generate error pattern matrix
    if K == 1:
        error_pat = np.eye(H.shape[1], dtype=int)
    else:
        error_pat = np.array(list(np.binary_repr(i, width=H.shape[1]) for i in range(2**H.shape[1])), dtype=int)

    print("error_pat",error_pat)

    syndromes = np.dot(error_pat, H) % 2

    return {tuple(syndrome): pattern for syndrome, pattern in zip(syndromes, error_pat)}

# Exercise 7
def decoder(G, H, Y):
    pass