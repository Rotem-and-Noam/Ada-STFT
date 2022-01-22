import os

from env_utils.check_points import LoadCkpt
from env_utils.options_parser import get_options
from env_utils.train_env import *

if __name__ == "__main__":

    # loading training options and hyper-parameters
    options = vars(get_options().parse_args())
    options["ckpt_dir"] = os.path.join(options["ckpt_dir"], options['test_name'])
    options["test"] = True

    print(f"Starting test: {options['test_name']}")

    # check if need to load check points
    ckpt = LoadCkpt(**options)
    ckpt_options = ckpt.load_options()

    # train
    data = 'val'
    env = Env(writer=None, ckpt=ckpt, options=options, **ckpt_options)
    model_accuracy, confusion_matrix, loss_total = env.calculate_accuracy_and_loss(data)
    env.show_confusion_matrix(confusion_matrix, model_accuracy, True)
    print(model_accuracy)

    print("done testing! Deep Learning Rules!")
