
from scipy import io
from string import ascii_lowercase as alphabets  # list of 26 letters
import numpy as np
import classify
import urllib2
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

WIDTH = 15  # Size of the board
DIMENSION = 10  # Reduced dimensions when applying PCA


# Trail 1
def extract_data(test):
    """Extract 450x450 pixels image and
    return 225x900 element matrix

    Keyword arguments:
    test -- matrix representing image
    """
    test_data = np.zeros((225, 900))
    i = 0    # row index of test_data array
    r = xrange(0, 450, 30)
    for row in r:
        for col in r:
            letter = test[row:row+30, col:col+30]    # 30x30 character
            test_data[i, :] = np.reshape(letter, (1, 900), order='F')
            i = i + 1
    return test_data


def search_vertical(word, board):
    """Search the board vertically from top to bottom
    and return position (row, col) which matches the most
    with the target word and that matching score value.

    Keyword arguments:
    word  -- labels list of target word
    board -- 15x15 letter labels matrix
    """
    l = len(word)  # length of target word
    matching = 0   # matching score
    row = 0
    col = 0
    for r in xrange(WIDTH-l+1):
        for c in xrange(WIDTH):
            w = board[r:r+l, c]
            s = (w == np.array(word)).tolist().count(True)  # count matching places
            if (s > matching):    # update if greater score found
                matching = s
                row = r
                col = c
    return row, col, matching


def search_diagonal(word, board):
    """Search the board diagonally from top left to bottom right
    and return position (row, col) which matches the most
    with the target word and that matching score value.

    Keyword arguments:
    word  -- labels list of target word
    board -- 15x15 letter labels matrix
    """
    l = len(word)   # length of target word
    matching = 0    # matching score
    row = 0
    col = 0
    ran = xrange(WIDTH-l+1)
    for r in ran:
        for c in ran:
            w = board[r:r+l, c:c+l].diagonal()
            s = (w == np.array(word)).tolist().count(True)  # count matching places
            if (s > matching):   # update if greater score found
                matching = s
                row = r
                col = c
    return row, col, matching


def search(word, board):
    """Search the board in 8 directions and
    return positions (row, col) of the first/last letters
    of target word in the board.

    Keyword arguments:
    word  -- labels list of target word
    board -- 15x15 letter labels matrix
    """
    l = len(word)-1
    INDEX = WIDTH - 1  # max index in the board

    # Each column stores the most matching positions for each direction
    score_lis = np.zeros((5, 8))

    r, c, s = search_vertical(word, board)  # top->bottom
    score_lis[:, 0] = [r, c, r+l, c, s]

    r, c, s = search_vertical(word, np.rot90(board, 2))  # bottom->top
    score_lis[:, 1] = [INDEX-r, INDEX-c, INDEX-(r+l), INDEX-c, s]

    r, c, s = search_vertical(word, board.transpose())  # left->right
    score_lis[:, 2] = [c, r, c, r+l, s]

    r, c, s = search_vertical(word, np.rot90(board))  # right->left
    score_lis[:, 3] = [c, INDEX-r, c, INDEX-(r+l), s]

    r, c, s = search_diagonal(word, board)  # top left->bottom right
    score_lis[:, 4] = [r, c, r+l, c+l, s]

    r, c, s = search_diagonal(word, np.rot90(board, 2))  # bottom right->top left
    score_lis[:, 5] = [INDEX-r, INDEX-c, INDEX-(r+l), INDEX-(c+l), s]

    r, c, s = search_diagonal(word, np.rot90(board, 3))  # bottom left->top right
    score_lis[:, 6] = [INDEX-c, r, INDEX-(c+l), r+l, s]

    r, c, s = search_diagonal(word, np.rot90(board, 1))  # top right->bottom left
    score_lis[:, 7] = [c, INDEX-r, c+l, INDEX-(r+l), s]

    max_index = np.argmax(score_lis[4, :])  # index of the largerst matching score

    return score_lis[:4, max_index]


def draw_lines(test, board, words, trial_name):
    """Draw coloured lines on board to indicate
    the positions of all target words

    Keyword arguments:
    test       -- matrix representing image
    board      -- 15x15 letter labels matrix
    trail_name -- name of trail (i.e. 1, 2)
    """
    word_lis = [w[0][0] for w in words]  # list of 24 target words
    plt.imshow(test, cmap=cm.Greys_r)
    plt.title(trial_name)

    for each_word in word_lis:
        w = [alphabets.index(i)+1 for i in each_word]  # labels list
        position = search(w, board)  # search target word in 8 directions
        [y1, x1, y2, x2] = 15 + 30*position  # coordinates of first/last letters
        plt.plot([x1, x2], [y1, y2], lw=1.5)  # line between first/last letters
    plt.show()


def wordsearch(test, words, train_data, train_labels):
    """Implement the search using 900 features for classification
    and plot the coloured lines to indicate word positions.

    Keyword arguments:
    test         -- matrix representing image
    words        -- array of 24 words
    train_data   -- training data set
    train_labels -- training data labels
    """
    test_data = extract_data(test)
    labels = classify.classify(train_data, train_labels, test_data)
    board = labels.reshape((WIDTH, WIDTH))  # 15x15 letter labels matrix
    draw_lines(test, board, words, "Trail 1")  # lines indicate words' positions


'''-------------------------------------------------------------------------------'''
# Trail 2
def reduce_dimension(train_data, test_data, d):
    """Reduce the dimensionality of train/test_data sets
    from 900 down to only d dimensions and
    return the new reduced train/test_data sets

    Keyword arguments:
    train_data -- the original training data set
    test_data  -- 225x900 test data set
    d          -- number of dimensions reduced to
    """
    covx = np.cov(train_data, rowvar=0)
    N = covx.shape[0]
    w, pc = scipy.linalg.eigh(covx, eigvals=(N-(d+1), N-2))  # starting from 2nd
    pc = np.fliplr(pc)
    delta_data = train_data - np.mean(train_data)  # center data at origin
    delta_test = test_data - np.mean(train_data)  # center test data at origin
    pcatest_data = np.dot(delta_test, pc)   # reduce test data
    pcatrain_data = np.dot(delta_data, pc)   # reduce train data
    return pcatrain_data, pcatest_data


def wordsearch_pca1(test, words, train_data, train_labels):
    """Implement the search using 10-dimension feature vectors
    and plot the coloured lines to indicate words' positions.

    Keyword arguments:
    test         -- matrix representing image
    words        -- array of 24 words
    train_data   -- data set used for training
    train_labels -- labels of training data
    """
    test_data = extract_data(test)
    pcatrain_data, pcatest_data = reduce_dimension(train_data, test_data, DIMENSION)
    labels = classify.classify(pcatrain_data, train_labels, pcatest_data)
    board = labels.reshape((WIDTH, WIDTH))  # 15x15 letter labels matrix
    draw_lines(test, board, words, "Trail 2")  # lines indicate words' positions


'''------------------------------------------------------------------------------'''
# Trail 3
def search_vertical_pca(word, board, mean_lis):
    """Search the board vertically from top to bottom
    and return position (row, col) which has minimum
    distance with the target word and that minimum distance value.

    Keyword arguments:
    word     -- labels list of target word
    board    -- 15x15x10 matrix representing test image
    mean_lis -- array of 26 mean vectors
    """
    l = len(word)  # word length
    dist = float("inf")   # initially set to +INFINITY
    row = 0
    col = 0
    for r in xrange(WIDTH-l+1):
        for c in xrange(WIDTH):
            w = board[r:r+l, c, :]   # 2d array
            d = 0
            for i, v in enumerate(word):
                a = w[i, :] - mean_lis[v-1, :]  # distance vector
                d = d + np.linalg.norm(a)   # compute overall distance

            if (d < dist):  # update distance if smaller value found
                dist = d
                row = r
                col = c
    return row, col, dist


def search_diagonal_pca(word, board, mean_lis):
    """Search the board diagonally from top left to bottom right
    and return row, column of position which has minimum
    distance with the target word and that distance.

    Keyword arguments:
    word     -- labels list of target word
    board    -- 15x15x10 matrix representing test image
    mean_lis -- array of 26 mean vectors
    """
    l = len(word)
    dist = float("inf")  # initially set to +INFINITY
    row = 0
    col = 0
    ran = xrange(WIDTH-l+1)
    for r in ran:
        for c in ran:
            w = board[r:r+l, c:c+l, :].diagonal().transpose()  # 2d array slice
            d = 0
            for i, v in enumerate(word):
                a = w[i, :] - mean_lis[v-1, :]  # distance vector
                d = d + np.linalg.norm(a)   # compute overall distance

            if (d < dist):  # update distance if smaller value found
                dist = d
                row = r
                col = c
    return row, col, dist


def search_pca(word, board, mean_lis):
    """Search the board in 8 directions and
    return the coordinates where has
    minimum distance to the target word.

    Keyword arguments:
    word     -- labels list of target word
    board    -- 15x15x10 matrix representing test image
    mean_lis -- array of mean vectors of 26 classes
    """
    l = len(word)-1
    INDEX = WIDTH - 1  # max index in 15x15 surface of the board

    # Each column stores the minimum-distance positions for each direction
    dist_lis = np.zeros((5, 8))

    # top->bottom
    r, c, d = search_vertical_pca(word, board, mean_lis)
    dist_lis[:, 0] = [r, c, r+l, c, d]

    # bottom->top
    r, c, d = search_vertical_pca(word, np.rot90(board, 2), mean_lis)
    dist_lis[:, 1] = [INDEX-r, INDEX-c, INDEX-(r+l), INDEX-c, d]

    # left->right
    r, c, d = search_vertical_pca(word, np.rot90(board, 3), mean_lis)
    dist_lis[:, 2] = [INDEX-c, r, INDEX-c, r+l, d]

    # right->left
    r, c, d = search_vertical_pca(word, np.rot90(board), mean_lis)
    dist_lis[:, 3] = [c, INDEX-r, c, INDEX-(r+l), d]

    # top left->bottom right
    r, c, d = search_diagonal_pca(word, board, mean_lis)
    dist_lis[:, 4] = [r, c, r+l, c+l, d]

    # bottom right->top left
    r, c, d = search_diagonal_pca(word, np.rot90(board, 2), mean_lis)
    dist_lis[:, 5] = [INDEX-r, INDEX-c, INDEX-(r+l), INDEX-(c+l), d]

    # bottom left->top right
    r, c, d = search_diagonal_pca(word, np.rot90(board, 3), mean_lis)
    dist_lis[:, 6] = [INDEX-c, r, INDEX-(c+l), r+l, d]

    # top right->bottom left
    r, c, d = search_diagonal_pca(word, np.rot90(board, 1), mean_lis)
    dist_lis[:, 7] = [c, INDEX-r, c+l, INDEX-(r+l), d]

    min_index = np.argmin(dist_lis[4, :])  # index of minimum distance value
    return dist_lis[:4, min_index]


def find_mean_lis(train_labels, pcatrain_data):
    """Find mean vectors of all 26 letters (i.e. classes)
    in the training data and return array of 26 mean vectors.

    Keyword arguments:
    train_labels  -- labels of training data
    pcatrain_data -- 10-dimensional training data
    """
    mean_lis = np.zeros((26, DIMENSION))
    labels = train_labels.transpose()

    for i in xrange(1, 27):
        data = pcatrain_data[labels[:, 0] == i, :]
        mean_lis[i-1, :] = np.mean(data, axis=0)  # mean of letter ith

    return mean_lis


def new_board(pcatest_data):
    """Create 3D board (15x15x10) to store all values
    from the image matrix in the same order and
    return that board.

    Keyword arguments:
    pcatest_data -- 10-dimensional test data
    """
    board = np.zeros((WIDTH, WIDTH, DIMENSION))  # 3d board
    i = 0
    ran = xrange(WIDTH)
    for r in ran:
        for c in ran:
            board[r, c, :] = pcatest_data[i, :]
            i = i+1
    return board


def wordsearch_pca2(test, words, train_data, train_labels):
    """Implement the search using 10-dimension feature vectors
    by measuring the distance to the target word
    and plot the coloured lines to indicate word positions.

    Keyword arguments:
    test         -- matrix representing image
    words        -- array of 24 words
    train_data   -- data set used for training
    train_labels -- labels of training data
    """
    test_data = extract_data(test)
    pcatrain_data, pcatest_data = reduce_dimension(train_data, test_data, DIMENSION)
    word_lis = [w[0][0] for w in words]  # list of 24 words
    mean_lis = find_mean_lis(train_labels, pcatrain_data)  # list of 26 mean vectors
    board = new_board(pcatest_data)   # 15x15x10 matrix reprenting test image
    plt.imshow(test, cmap=cm.Greys_r)
    plt.title("Trail 3")

    for each_word in word_lis:
        w = [alphabets.index(i)+1 for i in each_word]  # labels list
        position = search_pca(w, board, mean_lis)
        [y1, x1, y2, x2] = 15 + 30*position  # coordinates of first/last letters
        plt.plot([x1, x2], [y1, y2], lw=1.5)  # line between first/last letters
    plt.show()


def demo():
    """Load data (training, test) for the programme and
    run and display result for each Trail in turn
    """
    mat_dict = io.loadmat("assignment2.mat")
    train_data = mat_dict["train_data"].astype(np.float32)
    train_labels = mat_dict["train_labels"]
    test1 = mat_dict["test1"].astype(np.float32)
    test2 = mat_dict["test2"].astype(np.float32)
    words = mat_dict["words"]

    wordsearch(test1, words, train_data, train_labels)        # Trail 1
    wordsearch_pca1(test1, words, train_data, train_labels)   # Trail 2
    wordsearch_pca2(test2, words, train_data, train_labels)   # Trail 3


# Display the results
demo()
