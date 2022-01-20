import optuna
from codes.env_utils.train_env import Env
import json
from codes.env_utils.check_points import LoadCkpt
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt


def objective(trial):

    # loading training options and hyper-parameters
    with open("optuna_options.json", 'r') as fp1:
        optuna_options = json.load(fp1)

    optuna_options['learning_rate'] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optuna_options['optimizer_class'] = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    optuna_options['split_parts'] = trial.suggest_categorical("split_parts", [1, 3, 4, 6, 12])
    optuna_options['augmentation'] = trial.suggest_categorical("augmentation", [0, 1])
    optuna_options['gamma'] = trial.suggest_float("gamma", 0.99, 1)
    optuna_options['test_name'] = optuna_options['test_name'] + str(trial.number)
    optuna_options['ckpt_dir'] = os.path.join(optuna_options['ckpt_dir'] + optuna_options['test_name'])

    with open(os.path.join("logs", f"options_optuna{str(trial.number)}.json"), 'w') as fp2:
        json.dump(optuna_options, fp2)

    print(f"Starting test: {optuna_options['test_name']}")

    # check if need to load check points
    optuna_ckpt = LoadCkpt(**optuna_options)

    tensorboard_path = os.path.join(optuna_options['tensorboard_dir'], optuna_options['test_name'])
    optuna_writer = SummaryWriter(log_dir=tensorboard_path)

    # train
    optuna_env = Env(writer=optuna_writer, ckpt=optuna_ckpt, options=optuna_options, **optuna_options)

    val_accuracy = 0

    for epoch in range(optuna_env.start_epoch, optuna_env.epoch_num):

        train_loss = optuna_env.train_epoch()
        val_accuracy, confusion_matrix, val_loss = optuna_env.calculate_accuracy_and_loss()

        if val_accuracy > optuna_env.best_acc:
            optuna_env.best_acc = val_accuracy
            optuna_env.ckpt.save_ckpt(optuna_env.model, optuna_env.optimizer, optuna_env.scheduler,
                                      epoch, optuna_env.options, True)
            if optuna_env.writer is not None:
                optuna_env.writer.add_figure('best confusion matrix',
                                             optuna_env.show_confusion_matrix(confusion_matrix, val_accuracy),
                                             epoch)

        print(f"{optuna_env.test_name}: ",
              f"params: split: {optuna_options['split_parts']}, opt: {optuna_options['optimizer_class']},"
              f"aug: {optuna_options['augmentation']}   "
              f"epoch #{epoch}, val accuracy: {100 * val_accuracy:.4f}%",
              f"train loss: {train_loss:.5f}",
              f"val loss: {val_loss:.5f}",
              f"learning rate: {optuna_env.optimizer.param_groups[0]['lr']:.6f}")

        # send documentation to tensorboard
        if optuna_env.writer is not None:
            optuna_env.tensorboard_logging(confusion_matrix, train_loss, val_loss, val_accuracy, epoch)
        # save check points
        if epoch % optuna_env.ckpt_interval == optuna_env.ckpt_interval - 1:
            optuna_env.ckpt.save_ckpt(optuna_env.model, optuna_env.optimizer, optuna_env.scheduler,
                                      epoch, optuna_env.options)

        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy


if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="AdaSTFT", direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, gc_after_trial=True, n_jobs=4)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study Statistics: ")
    print("    Number of finished trials: ", len(study.trials))
    print("    Number of pruned trials: ", len(pruned_trials))
    print("    Number of complete trials: ", len(complete_trials))

    best = study.best_trial
    print(f"Best trial: {best.value}")

    print("    Params: ")
    for key, value in best.params.items():
        print(f"    {key}:{value}")

    with open("optuna_best_params.json", 'w') as fp:
        json.dump(best.params, fp)

    fig = optuna.visualizations.plot_param_importance(study)
    plt.savefig("optuna_plot_param_importance.png")
