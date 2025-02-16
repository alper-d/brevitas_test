import matplotlib.pyplot as plt

#pruning_amounts = ["0.3", "0.5", "0.6", "0.7", "0.8", "0.9"]
pruning_amounts = ["0.3", "0.5", "0.7", "0.9"]

train_logs = []
test_logs = []
for ratio in pruning_amounts:
    train_temp = []
    test_temp = []
    with open(f"run1/pruning_logs_{ratio}.txt", "r") as file1:
        lines = file1.readlines()
        for line in lines:
            if "complete. Train" in line:
                train_temp.append(float(line.split()[-1]))
            elif "complete. Test" in line:
                test_temp.append(float(line.split()[-1]))
    train_logs.append(train_temp)
    test_logs.append(test_temp)

# plt.figure(figsize=(8, 5))  # Set figure size
# for i, ratio in enumerate(pruning_amounts):
#    plt.plot(train_logs[i], label=f"Ratio={ratio}")
#
## Add labels and title
# plt.title("Train")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
#
## Add grid and legend
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
#
## Show the plot
# plt.show()
# plt.savefig()

plt.figure(figsize=(8, 5))  # Set figure size
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
plt.show()
