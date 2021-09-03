import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')

    parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
    parser.add_argument('-dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')
    parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')
    parser.add_argument('-init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
    parser.add_argument('-m_window', action='store', dest='m_window', help='Past history window.', type=int)
    parser.add_argument('-h_window', action='store', dest='h_window', help='Prediction window.', type=int)
    parser.add_argument('-end_window', action='store', dest='end_window', help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)
    parser.add_argument('-exp_folder', action='store', dest='exp_folder', help='Used when the dataset folder of the experiment is different than the default dataset.')
    parser.add_argument('-provided_videos', action="store_true", dest='provided_videos', help='Flag that tells whether the list of videos is provided in a global variable.')

    args = parser.parse_args()

    print(args)
    return args
