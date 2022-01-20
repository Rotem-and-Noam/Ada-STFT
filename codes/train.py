from env_utils.train_env import Env
from torch.utils.tensorboard import SummaryWriter
from env_utils.check_points import *
from env_utils.options_parser import get_options

if __name__ == "__main__":

    # loading training options and hyper-parameters
    options = vars(get_options().parse_args())
    options["ckpt_dir"] = os.path.join(options["ckpt_dir"], options['test_name'])

    print(f"Starting test: {options['test_name']}")

    # check if need to load check points
    ckpt = LoadCkpt(**options)
    if ckpt.start_epoch >= options['epoch_num']:
        print('This test is already done!')

    else:
        # tensorboard initialising
        tensorboard_path = os.path.join(options['tensorboard_dir'], options['test_name'])
        writer = SummaryWriter(log_dir=tensorboard_path)

        # train
        env = Env(writer=writer, ckpt=ckpt, options=options, **options)
        # env.calculate_accuracy_and_loss('val')
        env.train()

        print("done training! Deep Learning Rules!")
