import numpy as np
import matplotlib.pyplot as plt

def mean(r):
    out = []
    a = list(np.arange(0,len(r),50))
    for i in range(1,len(a)):
        out.append(np.mean(r[a[i-1]:a[i]]))
    return out

def specific_plot(r1, string,color):
    r1 = mean(r1)
    plt.scatter(range(len(r1)),r1,color = f'{color}')
    plt.xlabel('Episodes (in hundreds)')
    plt.ylabel('# Rewards')
    plt.title(f'{string}',fontsize=10)
    plt.ylim(-0.05,1.05)

def general_specific_plot(r1, r2, r3, r4):
    plt.suptitle(f'# Rewards vs Episodes | Sarsa | Frozen Lake', fontsize=13)
    plt.subplot(221)
    specific_plot(r1,'alpha=0.4, gamma=0.95','red')
    plt.subplot(222)
    specific_plot(r2,'alpha=0.2, gamma=0.95','yellow')
    plt.subplot(223)
    specific_plot(r3,'alpha=0.1, gamma=0.95','blue')
    plt.subplot(224)
    specific_plot(r4,'alpha=0.05, gamma=0.95','green')
    plt.tight_layout(pad=3.0)
    plt.savefig(f"Sarsa_FrozenLake_specific"+".jpg")  

def general_plot(r1,r2,r3,r4):
    r1,r2,r3,r4 = mean(r1),mean(r2),mean(r3),mean(r4)
    plt.scatter(range(len(r1)),r1,color = f'red', label='alpha=0.4')
    plt.scatter(range(len(r2)),r2,color = f'yellow', label='alpha=0.2')
    plt.scatter(range(len(r3)),r3,color = f'blue', label='alpha=0.1')
    plt.scatter(range(len(r4)),r4,color = f'green', label='alpha=0.05')
    plt.legend(loc='best')
    plt.xlabel('Episodes (in hundreds)')
    plt.ylabel('# Rewards')
    plt.suptitle(f'# Rewards vs Episodes | Sarsa | Frozen Lake',fontsize=13)
    plt.ylim(-0.05,1.05)
    plt.savefig(f"Sarsa_FrozenLake_general"+".jpg")     
    plt.close()