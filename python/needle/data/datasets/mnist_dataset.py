from typing import List, Optional, Iterable
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):  
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as img_file, gzip.open(label_filename, 'rb') as lbl_file:
            # Read magic numbers
            img_file.read(4)  # Skip magic number
            num_images = struct.unpack('>I', img_file.read(4))[0]
            num_rows = struct.unpack('>I', img_file.read(4))[0]
            num_cols = struct.unpack('>I', img_file.read(4))[0]

            lbl_file.read(8)  # Skip magic number and number of labels

            # Read image data and labels
            X = np.frombuffer(img_file.read(num_images * num_rows * num_cols), dtype=np.uint8)
            y = np.frombuffer(lbl_file.read(num_images), dtype=np.uint8)

            # Reshape and normalize image data
            X = X.reshape(num_images, num_rows * num_cols).astype(np.float32) / 255.0

        self.images = X
        self.label = y

    def __getitem__(self, index) -> object:
        
        if isinstance(index, (Iterable, slice)):
            img = [i.reshape(28, 28, 1) for i in self.images[index]]
        else:
            img = [self.images[index].reshape(28, 28, 1)]
        lbl = self.label[index]
        
        if self.transforms:
            for f in self.transforms:
                img = [f(i) for i in img]
        
        # why ?
        return [np.stack(img), lbl]

    def __len__(self) -> int:
        return len(self.images)

# class MNISTDataset(Dataset):
#     def __init__(
#         self,
#         image_filename: str,
#         label_filename: str,
#         transforms: Optional[List] = None,
#     ):
#         ### BEGIN YOUR SOLUTION
#         super().__init__(transforms)
#         self.images, self.labels = parse_mnist(
#             image_filesname=image_filename,
#             label_filename=label_filename
#         )
#         ### END YOUR SOLUTION

#     def __getitem__(self, index) -> object:
#         ### BEGIN YOUR SOLUTION
#         X, y = self.images[index], self.labels[index]
#         if self.transforms:
#             X_in = X.reshape((28, 28, -1))
#             X_out = self.apply_transforms(X_in)
#             return X_out.reshape(-1, 28 * 28), y
#         else:
#             return X, y
#         ### END YOUR SOLUTION

#     def __len__(self) -> int:
#         ### BEGIN YOUR SOLUTION
#         return self.labels.shape[0]
#         ### END YOUR SOLUTION

# class NDArrayDataset(Dataset):
#     def __init__(self, *arrays):
#         self.arrays = arrays

#     def __len__(self) -> int:
#         return self.arrays[0].shape[0]

#     def __getitem__(self, i) -> object:
#         return tuple([a[i] for a in self.arrays])
    

# def parse_mnist(image_filesname, label_filename):
#     """ Read an images and labels file in MNIST format.  See this page:
#     http://yann.lecun.com/exdb/mnist/ for a description of the file format.

#     Args:
#         image_filename (str): name of gzipped images file in MNIST format
#         label_filename (str): name of gzipped labels file in MNIST format

#     Returns:
#         Tuple (X,y):
#             X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
#                 data.  The dimensionality of the data should be
#                 (num_examples x input_dim) where 'input_dim' is the full
#                 dimension of the data, e.g., since MNIST images are 28x28, it
#                 will be 784.  Values should be of type np.float32, and the data
#                 should be normalized to have a minimum value of 0.0 and a
#                 maximum value of 1.0.

#             y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
#                 labels of the examples.  Values should be of type np.int8 and
#                 for MNIST will contain the values 0-9.
#     """
#     ### BEGIN YOUR SOLUTION
#     f = gzip.open(image_filesname)
#     data = f.read()
#     f.close()
#     h = struct.unpack_from('>IIII', data, 0)
#     offset = struct.calcsize('>IIII')
#     imgNum = h[1]
#     rows = h[2]
#     columns = h[3]
#     pixelString = '>' + str(imgNum * rows * columns) + 'B'
#     pixels = struct.unpack_from(pixelString, data, offset)
#     X = np.reshape(pixels, [imgNum, rows * columns]).astype('float32')
#     X_max = np.max(X)
#     X_min = np.min(X)
#     # X_max = np.max(X, axis=1, keepdims=True)
#     # X_min = np.min(X, axis=1, keepdims=True)
    
#     X_normalized = ((X - X_min) / (X_max - X_min))
    
  
#     f = gzip.open(label_filename)
#     data = f.read()
#     f.close()
#     h = struct.unpack_from('>II', data, 0)
#     offset = struct.calcsize('>II')
#     num = h[1]
#     labelString = '>' + str(num) + 'B'
#     labels = struct.unpack_from(labelString, data, offset)
#     y = np.reshape(labels, [num]).astype('uint8')
    
#     return (X_normalized,y)
#     ### END YOUR SOLUTION