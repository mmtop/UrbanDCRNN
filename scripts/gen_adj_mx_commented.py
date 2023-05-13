from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """

    # 1. Create an empty distance matrix and initialize all elements to infinity
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # 2. Create a dictionary to map sensor IDs to their corresponding indices in the distance matrix
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # 3. Fill the distance matrix using the input distance_df dataframe
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # 4. Calculate the standard deviation of the non-infinite distance values in the matrix
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    # 5. Compute the adjacency matrix using a Gaussian function
    # adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx = dist_mx
    # Optional: Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # 6. Set the adjacency values below the threshold to 0, effectively sparsifying the adjacency matrix
    # adj_mx[adj_mx < normalized_k] = 0

    # 7. Return the sensor IDs, the sensor ID-to-index mapping, and the final adjacency matrix
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    # 1. Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='data/sensor_graph/denhaag_sparse_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/sensor_graph/denhaag_cost.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/sensor_graph/adj_mat.pkl',
                        help='Path of the output file.')
    parser.add_argument('--node_number', type=int, default=207,
                        help='Number of nodes we want to use')
    
    # 2. Parse command line arguments
    args = parser.parse_args()

    # 3. Read the sensor IDs from the input file and limit the number of nodes
    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
        sensor_ids = sensor_ids[0:args.node_number]

    # 4. Read the distance data from the input CSV file
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})

    # 5. Compute the adjacency matrix using the distance data and sensor IDs
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, args.normalized_k)

    # 6. Save the sensor IDs, sensor ID-to-index mapping, and adjacency matrix to a pickle file
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
