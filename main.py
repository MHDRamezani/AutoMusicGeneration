import shutil

from train import *
from generate import *

DIRECTORY = 'D:/Business/Idea_Music/Data/Original_Data/IMSLP_GenMusicSeq2Seq/'
saved_params = 'D:/Business/Idea_Music/Code/GenerateMusic_Seq2Seq/saved_params'

COMPLEXITY_LEVEL_NUMBER = 12

if __name__ == '__main__':

    # -------------------------------------------------------------------------------------------
    # Loading Data + Feature Extraction
    # -------------------------------------------------------------------------------------------

    try:
        if os.path.isfile(saved_params) or os.path.islink(saved_params):
            os.unlink(saved_params)
        elif os.path.isdir(saved_params):
            shutil.rmtree(saved_params)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (saved_params, e))

    os.makedirs(saved_params)

    for cnt_complexity in range(1, 2):  # COMPLEXITY_LEVEL_NUMBER + 1):
        data_path = DIRECTORY + "{:02d}".format(cnt_complexity)
        model_path = "./saved_params/LSTM_model_" + "{:02d}".format(cnt_complexity) + ".h5"
        loss_plot = "./saved_params/TrainingLoss_" + "{:02d}".format(cnt_complexity) + ".png"

        print(cnt_complexity)
        train_model(data_path, model_path, loss_plot)

        sample_midi_path = 'D:/Business/Idea_Music/Data/Original_Data/IMSLP_GenMusicSeq2Seq/09/' \
                           'Albeniz Isaac - Espana, Op.165  No. 1. Preludio.mid'
        output_path = "./seq2seq_output/seq2seq_midi_" + "{:02d}".format(cnt_complexity) + ".mid"
        generate_midi(sample_midi_path, model_path, output_path)







