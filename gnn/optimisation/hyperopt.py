import os
import pickle
import time

import optuna
import torch

from gnn.loaders.load_ensembles2 import load_data as load_data_ensemble2
from gnn.loaders.load import load_data
from gnn.main import create_data, all_the_things, train_ensemble2
from gnn.networks.networks import create_network_two_no_conv_relu_dropout

EPOCHS = 250


def save(model, name):
    filename = f"model/{name}-tmp-model.pkl"
    with open(filename, "wb") as fp:
        pickle.dump(model, fp)
    return filename


def load(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def delete(path):
    os.remove(path)


def ensemble2_trainer(
        model,
        trial_no,
        data,
        optimizer,
        # loss_func,
        # should_prune=lambda _, __: False,
        l1_lambda=False,
        loss_type="min_loss_validation"
):

    def early_stopper(loss, epoch, cleanup):
        return
        # if should_prune(loss, epoch):
        #     cleanup()
        #     # shutil.rmtree(f"model/ensemble/{test_name}", ignore_errors=True)
        #     raise optuna.TrialPruned()
    retval = train_ensemble2(
        model,
        optimizer,
        data,
        f"optim-{trial_no}",
        False,
        EPOCHS,
        EPOCHS,
        -1,
        l1_lambda,
        l1_lambda is not False,
        0,
        False,
        early_stopper=early_stopper,
        print_epochs=False
    )

    return retval["basics"][loss_type]


def plain_trainer(model_cons, trial_no, data, optimizer_cons, loss_func, should_prune=lambda _, __: False, l1_lambda=False):
    no_improvement = 0
    test_loss = 100000000
    last_loss = test_loss
    test_name = f"{time.time()}"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_cons()
    optimizer = optimizer_cons(model)
    best_so_far = save(model, test_name)
    if not isinstance(data, list):
        data = [data]
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for batch_no, datum in enumerate(data):
            out = model((datum.x, datum.edge_index))
            model.eval()
            if len(datum.test.x) > 0:
                out_test = model((datum.test.x, datum.test.edge_index))
                test_y = datum.test.y
            else:
                out_test = model((datum.x, datum.edge_index))
                test_y = datum.y

            loss = loss_func(out, datum.y)

            if l1_lambda is not False:
                l1_parameters = []
                for parameter in model.parameters():
                    l1_parameters.append(parameter.view(-1))
                loss = loss + l1_lambda * torch.abs(torch.cat(l1_parameters)).sum()

            loss.backward()
            optimizer.step()
            test_loss = (float(loss_func(out_test, test_y)))

            if should_prune(test_loss, epoch):
                delete(best_so_far)
                raise optuna.TrialPruned()

            if test_loss < last_loss:
                no_improvement = 0
                delete(best_so_far)
                best_so_far = save(model, test_name)
            else:
                no_improvement += 1
            last_loss = test_loss

    model = load(best_so_far)

    model.eval()

    if len(datum.test.x) > 0:
        out_test = model((datum.test.x, datum.test.edge_index))
        test_y = datum.test.y
    else:
        out_test = model((datum.x, datum.edge_index))
        test_y = datum.y
    test_loss = (float(loss_func(out_test, test_y)))
    delete(best_so_far)

    return test_loss


def plain_trainer_no_edge(model_cons, trial_no, data, optimizer_cons, loss_func, should_prune=lambda _, __: False, l1_lambda=False):
    no_improvement = 0
    test_loss = 100000000
    last_loss = test_loss
    test_name = f"{time.time()}"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_cons()
    optimizer = optimizer_cons(model)
    best_so_far = save(model, test_name)
    if not isinstance(data, list):
        data = [data]
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for batch_no, datum in enumerate(data):
            out = model(datum.x)
            model.eval()
            if len(datum.test.x) > 0:
                out_test = model(datum.test.x)
                test_y = datum.test.y
            else:
                out_test = model(datum.x)
                test_y = datum.y
            loss = loss_func(out, datum.y)
            #
            if l1_lambda is not False:
                l1_parameters = []
                for parameter in model.parameters():
                    l1_parameters.append(parameter.view(-1))
                loss = loss + l1_lambda * torch.abs(torch.cat(l1_parameters)).sum()

            loss.backward()
            optimizer.step()
            test_loss = (float(loss_func(out_test, test_y)))
            if should_prune(test_loss, epoch):
                delete(best_so_far)
                raise optuna.TrialPruned()

            if test_loss < last_loss:
                no_improvement = 0
                delete(best_so_far)
                best_so_far = save(model, test_name)
            else:
                no_improvement += 1
            last_loss = test_loss

    model = load(best_so_far)

    model.eval()
    if len(datum.test.x) > 0:
        out_test = model(datum.test.x)
        test_y = datum.test.y
    else:
        out_test = model(datum.x)
        test_y = datum.y
    test_loss = (float(loss_func(out_test, test_y)))
    delete(best_so_far)

    return test_loss


def save_output(name, output):
    name = f"optuna/{name}.csv"
    keys = sorted(output.keys())
    if not os.path.exists(name):
        with open(name, "w") as fp:
            fp.write(",".join(keys) + "\n")

    with open(name, "a") as fp:
        fp.write(",".join(f'"{output[k]}"' if isinstance(k, str) else f"{output[k]}" for k in keys) + "\n")


def create_objective(
        trainer,
        optimizer,
        loader,
        model_cons,
        mparams=None,
        lparams=None,
        oparams=None,
        train_name=None,
        **kwargs
):
    if train_name is None:
        train_name = f"{time.time()}"
    if mparams is None:
        mparams = {}
    if lparams is None:
        lparams = {}
    if oparams is None:
        oparams = {}

    trial_no = -1

    data, num_features = create_data(loader, params=lparams.copy(), plot=False)

    def objective(trial):
        nonlocal trial_no, data, optimizer
        local_optimizer = optimizer

        trainer_params = {}
        for key, item in kwargs.items():
            if isinstance(item, tuple):
                trainer_params[key] = getattr(trial, f"suggest_{item[0]}")(key, *item[1:])
            else:
                trainer_params[key] = item
        model_params = {"inp_size": num_features}
        for key, item in mparams.items():
            if isinstance(item, tuple):
                model_params[key] = getattr(trial, f"suggest_{item[0]}")(key, *item[1:])
            else:
                model_params[key] = item

        optimizer_params = {}
        for key, item in oparams.items():
            if isinstance(item, tuple):
                optimizer_params[key] = getattr(trial, f"suggest_{item[0]}")(key, *item[1:])
            else:
                optimizer_params[key] = item

        model = model_cons(**model_params)

        if isinstance(local_optimizer, list):
            local_optimizer = trial.suggest_categorical('optimizer', local_optimizer)

        local_optimizer = local_optimizer(model, **optimizer_params)
        loss_func = torch.nn.MSELoss()

        def should_prune(loss, epoch):
            trial.report(loss, epoch)
            return trial.should_prune()

        trial_no += 1
        loss = trainer(model, trial_no, data, local_optimizer, loss_func, should_prune=should_prune, **trainer_params)
        output = {
            **lparams,
            **model_params,
            **optimizer_params,
            **trainer_params,
            "loss": loss,
            "trial_no": trial_no,
            "trainer": trainer.__name__,
            "model": model.__name__
        }
        save_output(train_name, output)
        return loss
    return objective


def model_constructor_constructor(model_c):
    def model_constructor(*mc_args, **mc_kwargs):
        def model(*m_args, **m_kwargs):
            return model_c(*(mc_args + m_args), **{**mc_kwargs, **m_kwargs})
        model.__name__ = model_c.__name__
        return model
    return model_constructor


def optimizer_constructor(optim):
    def optimizer(model, **m_kwargs):
        return optim(model.parameters(), **m_kwargs)
    return optimizer


def optimizer_constructor_constructor(optim):
    def internal_optimizer_constructor(_, **mc_kwargs):
        def optimizer(model, **m_kwargs):
            return optim(model.parameters(), **{**mc_kwargs, **m_kwargs})
        return optimizer
    return internal_optimizer_constructor


if __name__ == "__main__":
    tests = {
        "trainer": [ensemble2_trainer],
        "optimizer": [torch.optim.Adam],
        "model": [create_network_two_no_conv_relu_dropout],
        "loader": [load_data_ensemble2],
        "dropout": [("float", 0, 0.5)],
        "lr": [("float", 0.002, 0.01)],
        "weight_decay": [("float", 2.30038546822231E-05, 3.30038546822231E-05)],
        "filename": ["WHEAT_combined.csv", ],
        "algorithm": ["euclidean"],
        "split": [8],
        "use_weights": [True],
        "l1_lambda": [("float", 0.000011, 0.001)],
        "loss_type": ["min_loss_validation"],
        "separate_sets": [True],
        "use_validation": [True],
        "use_l1_reg": [True],
    }

    num_trials = 2500
    for params in all_the_things(tests):
        study = optuna.create_study(direction="minimize")
        try:
            study.optimize(
                create_objective(
                    params.get("trainer", plain_trainer),
                    optimizer_constructor_constructor(params.get("optimizer", torch.optim.Adam)),
                    params.get("loader", load_data),
                    model_constructor_constructor(params.get("model", create_network_two_no_conv_relu_dropout)),
                    mparams={
                        "out_size": params.get("out_size", 1),
                        "conv_kernel_size": params.get("conv_kernel_size", 10),
                        "filters": params.get("filters", 20),
                        "pool_size": params.get("pool_size", 2),
                        "internal_size": params.get("internal_size", 100),
                        "dropout": params.get("dropout", ("float", 0.4, 0.5)),
                    },
                    oparams={
                        "lr": params.get("lr", ("float", 0.00002, 0.003)),
                    },
                    lparams={
                        "filename": params.get("filename", "WHEAT1.csv"),
                        "num_neighbours": params.get("num_neighbours", 3),
                        "smoothing": params.get("smoothing", "laplacian"),
                        "mode": params.get("mode", "distance"),
                        "split": params.get("split", 0.8),
                        "algorithm": params.get("algorithm", "euclidean"),
                        "use_weights": params.get("use_weights", True),
                        "separate_sets": params.get("separate_sets", True),
                        "use_validation": params.get("use_validation", True),
                    },
                    l1_lambda=params.get("l1_lambda", ("float", 1.112E-05, 2.112E-05)),
                    loss_type=params.get("loss_type", "min_loss_validation")

                ),
                n_trials=num_trials)
        except Exception as e:
            print(e)

