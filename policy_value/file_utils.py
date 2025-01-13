import os
import wandb


def initialize_wandb(rl_model, args, n_playout=None, epsilon=None, quantiles=None, effi_n_playout=None):
    common_config = {
        "entity": "hails",
        "project": "gym_4iar_sh2",
        "config": args.__dict__
    }

    if rl_model == "DQN":
        run_name = f"FIAR-{rl_model}-MCTS{n_playout}-Eps{epsilon}"
    elif rl_model == "QRDQN":
        run_name = f"FIAR-{rl_model}-MCTS{n_playout}-Quantiles{quantiles}-Eps{epsilon}"
    elif rl_model in ["AC", "QAC"]:
        run_name = f"FIAR-{rl_model}-MCTS{n_playout}"
    elif rl_model in ["QRAC", "QRQAC"]:
        run_name = f"FIAR-{rl_model}-MCTS{n_playout}-Quantiles{quantiles}"
    elif rl_model == "EQRDQN":
        run_name = f"FIAR-{rl_model}-MCTS{effi_n_playout}-Eps{epsilon}"
    elif rl_model == "EQRQAC":
        run_name = f"FIAR-{rl_model}-MCTS{effi_n_playout}"
    else:
        raise ValueError("Model is not defined")

    wandb.init(name=run_name, **common_config)


def create_models(rl_model, epsilon=None, n_playout=None, quantiles=None, effi_n_playout=None, i=None):
    """
    Generate training and evaluation model file paths dynamically based on the rl_model and parameters.
    """
    base_paths = {
        "Training": "Training",
        "Eval": "Eval"
    }

    # Define model-specific path structures
    model_params = {
        "DQN": f"_nmcts{n_playout}_eps{epsilon}",
        "QRDQN": f"_nmcts{n_playout}_quantiles{quantiles}_eps{epsilon}",
        "EQRDQN": f"_nmcts{effi_n_playout}_eps{epsilon}",
        "AC": f"_nmcts{n_playout}",
        "QAC": f"_nmcts{n_playout}",
        "QRAC": f"_nmcts{n_playout}_quantiles{quantiles}",
        "QRQAC": f"_nmcts{n_playout}_quantiles{quantiles}",
        "EQRQAC": f"_nmcts{effi_n_playout}"
    }

    if rl_model not in model_params:
        raise ValueError("Model is not defined")

    # Construct the specific path part for the model
    specific_path = model_params[rl_model]
    filename = f"train_{i + 1:03d}.pth"

    # Generate full paths
    model_file = f"{base_paths['Training']}/{rl_model}{specific_path}/{filename}"
    eval_model_file = f"{base_paths['Eval']}/{rl_model}{specific_path}/{filename}"

    return model_file, eval_model_file


def get_existing_files(rl_model, n_playout=None, epsilon=None, quantiles=None, effi_n_playout=None):
    """
    Retrieve a list of existing file indices based on the model type and parameters.
    """
    base_path = "Training"
    if rl_model == "DQN":
        path = f"{base_path}/{rl_model}_nmcts{n_playout}_eps{epsilon}"
    elif rl_model == "QRDQN":
        path = f"{base_path}/{rl_model}_nmcts{n_playout}_quantiles{quantiles}_eps{epsilon}"
    elif rl_model == "EQRDQN":
        path = f"{base_path}/{rl_model}_nmcts{effi_n_playout}_eps{epsilon}"
    elif rl_model in ["AC", "QAC"]:
        path = f"{base_path}/{rl_model}_nmcts{n_playout}"
    elif rl_model in ["QRAC", "QRQAC"]:
        path = f"{base_path}/{rl_model}_nmcts{n_playout}_quantiles{quantiles}"
    elif rl_model == "EQRQAC":
        path = f"{base_path}/{rl_model}_nmcts{effi_n_playout}"
    else:
        raise ValueError("Model is not defined")

    # Fetch files and extract indices
    return [
        int(file.split('_')[-1].split('.')[0])
        for file in os.listdir(path)
        if file.startswith('train_')
    ]