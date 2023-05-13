import argparse
import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor_eval import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        mean_score, outputs, mape_score, rmse_score, mae_std, mape_std, rmse_std = supervisor.evaluate('test')
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {} ({})".format(mean_score, mae_std))
        print("RMSE : {} ({})".format(rmse_score, rmse_std))
        print("MAPE : {} ({})".format(mape_score, mape_std))
        print('Predictions saved as {}.'.format(args.output_filename))
        # return {'MAE': mean_score, 'RMSE': rmse_score, 'MAPE': mape_score}


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)