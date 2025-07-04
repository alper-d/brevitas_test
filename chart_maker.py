import matplotlib.pyplot as plt

# pruning_amounts = ["0.3", "0.5", "0.6", "0.7", "0.8", "0.9"]
pruning_amounts = ["0.5", "0.7", "0.8", "0.9"]
pruning_amounts = ["0.5"]
train_logs = []
test_logs = []
run_folder = "run_2w2a_4layer"
experiment_time = ""
prefix_for_time = "Starting to write at "
for ratio in pruning_amounts:
    train_temp = []
    test_temp = []
    with open(f"./{run_folder}/pruning_logs.txt", "r") as file1:
        lines = file1.readlines()
        for line in lines:
            if prefix_for_time in line:
                experiment_time = line.removeprefix(prefix_for_time)
                experiment_time = experiment_time.split("on")
                experiment_time = experiment_time[0] + experiment_time[1].replace(
                    " ", "_"
                ).removesuffix("\n")
            elif "complete. Train" in line:
                train_temp.append(float(line.split()[-1]))
            elif "complete. Test" in line:
                test_temp.append(float(line.split()[-1]))
    train_logs.append(train_temp)
    test_logs.append(test_temp)

figure_size = (16, 10)
plt.figure(figsize=figure_size)  # Set figure size
for i, ratio in enumerate(pruning_amounts):
    plt.plot(train_logs[i], label=f"Ratio={ratio}")

# Add labels and title
plt.title("Train")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Add grid and legend
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Show the plot
# plt.show()
plt.savefig(f"./{run_folder}/train_{experiment_time}.png")

plt.figure(figsize=figure_size)  # Set figure size
for i, ratio in enumerate(pruning_amounts):
    plt.plot(test_logs[i], label=f"Ratio={ratio}")

# Add labels and title
plt.title("Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Add grid and legend
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Show the plot
# plt.show()
plt.savefig(f"./{run_folder}/test_{experiment_time}.png")
