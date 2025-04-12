import numpy as np

# Copy from Sintel Depth Dataset official I/O Python codes
# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


def cam2txt(cam_filepath, output_path, baseline=0.1, verbose=True):
    '''
    Function:
        Transform the Sintel .cam format camera parameters to .txt format,
        which is used for Foundation Stereo

    :param cam_filepath: the filepath of Sintel .cam format camera parameters file
    :param output_path: the output filepath of .txt format camera parameters file
    :param baseline: the length of baseline in the unit of meter
    :param verbose: whether to print the information of transformation

    :return:
        a dict which contains the intrinsic and extrinsic matrix and baseline
    '''

    f = open(cam_filepath, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    M = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
    N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))

    with open(output_path, 'w') as K:
        K.write(' '.join(map(str, M.flatten())) + '\n')
        K.write(str(baseline) + '\n')
        K.write(' '.join(map(str, M.flatten())) + '\n')

    if verbose:
        print(f"The intrinsic and extrinsic matrix have been saved to {output_path}")

    return {'M': M, 'N':N, 'baseline':baseline}

print(1)



