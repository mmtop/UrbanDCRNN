import yaml
import os 
import sys
import argparse

if __name__=="__main__":
    tuning_dic={"horizon":[8]}
    
    #possible things to check for hyperparambeter tuning
    # if mutliple things we could make dict 
    # key = name(lr, epochs, dropout) 
    # value = list of values
    # "num_nodes":[35,50] #max 207
    #learning_rates=[0.01,0.001] #max 0.01
    #epochs=[20,50] #max 100
    #dropout=[0.1,0.2] #max 1
    
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_dh.yaml', type=str,
                        help='Config file for pretrained model.')
    args = parser.parse_args()
    config = None

    for key in tuning_dic:
        for value in tuning_dic[key]:

            with open(args.config_filename) as f:
                config = yaml.safe_load(f)

            config['model'][key]=value
            
            with open(args.config_filename,"w") as f:
                yaml.dump(config,f)
            
            # os.system("python gen_data_sparse.py")
            os.system("python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_dh.yaml")