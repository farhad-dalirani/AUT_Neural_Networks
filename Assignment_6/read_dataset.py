def read_dataset(ith_col=0):
    """
    Read ith column of input dataset which is located in 'dataset\synthetic.data\synthetic'
    :param ith_col:
    :return:
    """
    import numpy as np

    with open('dataset\synthetic.data\synthetic') as f:
        lines = f.readlines()

    col = []

    # read specified column
    for i in range(0, len(lines)):

        # clean string
        str = lines[i]

        # break a line of numbers by whitespace
        different_cols_str = str.split()

        # convert string to number of specified column
        col.append(float(different_cols_str[ith_col]))

    col = np.array(col, dtype=np.float64)
    col = np.reshape(col, (col.shape[0], 1))

    return col


if __name__ == '__main__':
    col = read_dataset(ith_col=0)

    print(col)

    print(col.shape)
    print(col[0:10000].shape)
