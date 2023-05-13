# Just run this using python gen_data.py 

import argparse
import os
import sys
import yaml

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_dh.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--sensor_ids_filename', default = 'data/sensor_graph/graph_sensor_ids.txt', type=str,
                        help='File with the ids of selected sensors')
    parser.add_argument('--sensor_indices_filename', default = 'data/sensor_graph/graph_indices.txt', type=str,
                        help='File with the indices of selected sensors')
    args = parser.parse_args()
    
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)

        num_nodes = config['model'].get('num_nodes')
        horizon = config['model'].get('horizon')
        seq_len = config['model'].get('seq_len')

        os.system("mkdir -p data/DH")

        os.system(f"python -m scripts.gen_adj_mx  --sensor_ids_filename={args.sensor_ids_filename} --normalized_k=0.1\
        --output_pkl_filename=data/sensor_graph/adj_mx.pkl --node_number={num_nodes}")

        os.system(f"python -m scripts.generate_training_data --output_dir=data/DH --traffic_df_filename=data/DH.h5 \
                --node_number={num_nodes} --horizon={horizon} --seq_len={seq_len} --sensor_index_filename={args.sensor_indices_filename}")
        