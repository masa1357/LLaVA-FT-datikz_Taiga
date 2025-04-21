import matplotlib.pyplot as plt
import ast


def parse_log_file(filepath):
    train_logs = []
    eval_logs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    entry = ast.literal_eval(line)
                    if 'eval_loss' in entry:
                        eval_logs.append(entry)
                    elif 'loss' in entry:
                        train_logs.append(entry)
                except Exception as e:
                    print(f"Failed to parse line: {line}\n{e}")
    return train_logs, eval_logs

def plot_logs(train_logs, eval_logs):
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # --- Training ---
    epochs_train = [e['epoch'] for e in train_logs]
    losses = [e['loss'] for e in train_logs]
    grad_norms = [e['grad_norm'] for e in train_logs]
    lrs = [e['learning_rate'] for e in train_logs]

    axs[0].plot(epochs_train, losses, label='Training Loss')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss")
    axs[0].grid(True)

    axs[1].plot(epochs_train, grad_norms, label='Grad Norm', color='orange')
    axs[1].set_ylabel("Grad Norm")
    axs[1].set_title("Gradient Norm")
    axs[1].grid(True)

    axs[2].plot(epochs_train, lrs, label='Learning Rate', color='green')
    axs[2].set_ylabel("Learning Rate")
    axs[2].set_title("Learning Rate")
    axs[2].grid(True)

    # --- Evaluation ---
    if eval_logs:
        epochs_eval = [e['epoch'] for e in eval_logs]
        eval_losses = [e['eval_loss'] for e in eval_logs]
        axs[3].plot(epochs_eval, eval_losses, label='Eval Loss', color='red', marker='o')
        axs[3].set_ylabel("Eval Loss")
        axs[3].set_title("Evaluation Loss")
        axs[3].grid(True)

    axs[3].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("plot_train_log.png")
    plt.savefig("plot_train_log.pdf")

# 実行
if __name__ == "__main__":
    log_path = '/home/taiga/experiments/LLaVa-FT-datikz/wandb/run-20250421_131255-xbozgn20/files/output.log'
    train_logs, eval_logs = parse_log_file(log_path)
    plot_logs(train_logs, eval_logs)
