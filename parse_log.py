"""
Parse logs to retroactively create graphs of training history
"""

import matplotlib.pyplot as plt
import parse

LOG_NAME = "log4.txt"

curr_epoch = 1
curr_batch = 1

all_loss = []
all_acc = []
all_f1 = []

epoch_loss = []
epoch_acc = []
epoch_f1 = []

val_loss = []
val_acc = []
val_f1 = []

for line in open(LOG_NAME, 'r'):
    if "loss:" in line:
        # batch complete
        # loss: 0.0918 - acc: 0.9571 - f1: 0.7729
        try:
            loss = parse.search('loss: {:f}', line)[0]
            acc = parse.search('acc: {:f}', line)[0]
            f1 = parse.search('f1: {:S}', line)[0]
            f1 = 0 if f1 == 'nan' else float(f1)
        except:
            continue
            # some formatting issue happened :(

        all_loss.append(loss)
        all_acc.append(acc)
        all_f1.append(f1)

        if "val_loss:" in line:
        # FULL EPOCH COMPLETE
        # loss: 0.0918 - acc: 0.9570 - f1: 0.7731 - val_loss: 0.0778 - val_acc: 0.9602 - val_f1: nan
            v_loss = parse.search('val_loss: {:f}', line)[0]
            v_acc = parse.search('val_acc: {:f}', line)[0]
            v_f1 = parse.search('val_f1: {:S}', line)[0]
            v_f1 = 0 if v_f1 == 'nan' else float(v_f1)

            val_loss.append(v_loss)
            val_acc.append(v_acc)
            val_f1.append(v_f1)

            epoch_loss.append(loss)
            epoch_acc.append(acc)
            epoch_f1.append(f1)


plt.title('Training Loss per Batch')
plt.plot(all_loss, 'b')
plt.xlabel('Batches')
plt.ylabel('Training Loss')
plt.savefig("result_all.png")

plt.clf()
plt.title('Training Accuracy and F1')
plt.plot(all_acc, 'r')
plt.plot(all_f1, 'g')
plt.xlabel('Batches')
plt.savefig("result_all_acc.png")

plt.clf()
plt.title('Training Loss per Epoch')
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.plot(epoch_loss, 'b')
plt.savefig("result_epoch.png")

plt.clf()
plt.title('Training Accuracy and F1')
plt.plot(epoch_acc, 'r')
plt.plot(epoch_f1, 'g')
plt.xlabel('Epochs')
plt.savefig("result_epoch_acc.png")

plt.clf()
plt.title('Validation Loss per Epoch')
plt.plot(val_loss, 'b')
plt.ylabel('Validation Loss')
plt.xlabel('Epochs')
plt.savefig("result_val_acc.png")

plt.clf()
plt.title('Validation Accuracy and F1')
plt.plot(val_acc, 'r')
plt.plot(val_f1, 'g')
plt.xlabel('Epochs')
plt.savefig("result_val.png")



print(val_acc)
print(val_f1)



