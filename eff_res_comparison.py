import matplotlib.pyplot as plt
import pandas as pd

df_eff = pd.read_csv("csv/stats_efficientnet.csv")
df_res = pd.read_csv("csv/stats_resnet.csv")

eff_train_loss = df_eff['train_loss']
eff_train_acc = df_eff['train_acc']
eff_val_loss = df_eff['val_loss']
eff_val_acc = df_eff['val_acc']

res_train_loss = df_res['train_loss']
res_train_acc = df_res['train_acc']
res_val_loss = df_res['val_loss']
res_val_acc = df_res['val_acc']

#fig = plt.figure(1, figsize=(6,6))

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6,6), sharey=True)

ax1, ax2 = axs[0]
ax3, ax4 = axs[1]
x_ticks = range(len(eff_train_loss))
show_ticks = [0, 3, 6, 9, 12]

ax1.set_title("Train Loss")
ax1.plot(eff_train_loss, color='b')
ax1.plot(res_train_loss, color='r')
ax1.set_xlim(0,len(eff_train_acc)-1)
ax1.set_ylim(0,1)
ax1.set_xticks([x_ticks[i] for i in show_ticks])
ax1.set_ylabel("Loss")


ax2.set_title("Val Loss.")
ax2.plot(eff_val_loss, color='b')
ax2.plot(res_val_loss, color='r')
ax2.set_xlim(0,len(eff_train_acc)-1)
ax2.set_ylim(0,1)
ax2.set_xticks([x_ticks[i] for i in show_ticks])


ax3.set_title("Train Acc.")
ax3.plot(eff_train_acc, color='b')
ax3.plot(res_train_acc, color='r')
ax3.set_xlim(0,len(eff_train_acc)-1)
ax3.set_ylim(0,1)
ax3.set_xticks([x_ticks[i] for i in show_ticks])
ax3.set_ylabel("Accuracy")
ax3.set_xlabel("Epoch")

ax4.set_title("Val Acc.")
ax4.plot(eff_val_acc, color='b')
ax4.plot(res_val_acc, color='r')
ax4.set_xlim(0,len(eff_train_acc)-1)
ax4.set_ylim(0,1)
ax4.set_xticks([x_ticks[i] for i in show_ticks])
ax4.set_xlabel("Epoch")
plt.tight_layout()
plt.show()