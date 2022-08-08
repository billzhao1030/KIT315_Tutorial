from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

class MNIST(object):
    def __init__(self,dat_dir):
        self.mnist = input_data.read_data_sets(dat_dir, one_hot=True)

    
    def samples(self,batch_size=1):
        x,y = self.mnist.train.next_batch(batch_size)
        return x

    @property
    def dimension(self):
        return  784

    def show_samples(self,samples):
        n = math.floor(math.sqrt(samples.shape[0]))
        fig = plt.figure(figsize=(n, n))
        gs = gridspec.GridSpec(n,n )
        gs.update(wspace=0.05, hspace=0.05)
        
        for i in range(n*n):
            sample = samples[i]
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        plt.savefig('./vis.png', bbox_inches='tight')
        plt.close(fig)
    
        
