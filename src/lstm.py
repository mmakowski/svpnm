import numpy as np # type: ignore
import pandas as pd # type: ignore


def make_batches(dataset, batch_size, backprop_steps, pad_code, random_state):
    """
    Format the data into chunks `[[<batch_size> x <backprop_steps>]]`. A single token would be addressed by `input_batches[batch][chunk][document][token]`. Target labels are also split into the same batches, with a label for a given document accessed through `target_batches[batch][document]`. All documents within a single batch are right-padded with the PAD code so that all are of equal length that is a multiple of `backprop_steps`. The number of chunks can differ between batches, and is determined by the length of the longest document in the batch. The number of documents must be a multiple of batch size; the examples from the majority class (negatives in our case) will be subsampled to achieve this.

    For example, with batch size 2, backpropagation steps 3, and the following input documents and target labels:

    ```
    {
        [ 1,  2,  3,  4,  5,  6,  7] : 0,
        [ 8,  9, 10, 11]             : 1,
        [12, 13, 14, 15, 16]         : 0,
        [17, 18]                     : 0
    }
    ```

    the `input_batches` will look as follows:

    ```
    [
        # batch 1
        [ 
            # chunk 1
            [[1, 2,  3],
             [8, 9, 10]],
            # chunk 2
            [[ 4, 5, 6],
             [11, _, _]],
            # chunk 3
            [[7, _, _],
             [_, _, _]]
        ],
        # batch 2
        [
            # chunk 1
            [[12, 13, 14],
             [17, 18,  _]],
            # chunk 2
            [[15, 16, _],
             [ _,  _, _]]
        ]
    ]
    ```

    where `_` indicates the padding, and the `target_batches` will look as follows:

    ```
    [ 
        # batch 1
        [[0]
         [1]],
        # batch 2
        [[0],
         [0]]
    ]
    ```
    """
    dataset_trimmed = _adjust_set_size_to_batch_size(dataset, batch_size, random_state)
    assert len(dataset_trimmed) % batch_size == 0
    dataset = dataset_trimmed.sample(frac=1, random_state=random_state)
    n_batches = len(dataset) // batch_size
    for i_batch in range(0, n_batches):
        batch_examples = dataset[i_batch * batch_size : (i_batch+1) * batch_size]
        target_batch = np.array(batch_examples['label'], ndmin=2).T
    
        max_example_length_tokens = max(batch_examples['vectors'].apply(len))
        batch_length_chunks = int(np.ceil(max_example_length_tokens / backprop_steps))
        batch_length_tokens = batch_length_chunks * backprop_steps
        batch_tokens = _pad(batch_examples['vectors'], batch_length_tokens, pad_code)
        input_batch = np.array(np.hsplit(batch_tokens, batch_length_chunks))
        assert input_batch.shape == (batch_length_chunks, batch_size, backprop_steps)
        yield input_batch, target_batch


def _pad(vectors_collection, required_length, pad_code):
    """Pads all lists of vectors (represented as one-hot index) in the collection with the pad vector.
       Returns NumPy array.
    """
    return np.array([vectors + [pad_code] * (required_length - len(vectors)) 
                     for vectors in vectors_collection])


def _adjust_set_size_to_batch_size(dataset, batch_size, random_state):
    """Trims the number of samples so that it is a multiple of batch size. Only removes negatives.
       Return the trimmed input dataset.
    """
    example_count = len(dataset)
    required_count = example_count - (example_count % batch_size)
    positives = dataset[dataset['label'] == 1]
    negatives = dataset[dataset['label'] == 0]
    required_negatives = len(negatives) - (example_count - required_count)
    sampled_negatives = negatives.sample(n=required_negatives, random_state=random_state)
    dataset = pd.concat([positives, sampled_negatives]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return dataset
