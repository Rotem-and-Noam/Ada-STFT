from train_env import *

if __name__ == "__main__":

    # loading training options and hyper-parameters
    options = vars(get_options().parse_args())
    options["ckpt_dir"] = os.path.join(options["ckpt_dir"], options['test_name'])

    print(f"Starting test: {options['test_name']}")

    # check if need to load check points
    ckpt = LoadCkpt(**options)

    # train
    data = 'test'
    env = Env(writer=writer, ckpt=ckpt, options=options, **options)
    env.calculate_accuracy_and_loss(data)

    print("done testing! Deep Learning Rules!")

