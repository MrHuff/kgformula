import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    one = np.random.randn(10000)
    two = np.random.randn(10000)*10

    plt.hist(one,bins=100)
    plt.hist(two,bins=100,alpha=0.3)
    plt.show()