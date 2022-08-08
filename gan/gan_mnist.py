from gan import GAN
from mnist import MNIST

class Config():
    lr = 0.0001
    z_dimension =  784
    gen_hid_num = 100
    dis_hid_num = 100
    batch_size  = 100
    k = 1
if __name__=="__main__":
    dataset = MNIST("./data/MNIST/")
    conf = Config()
    
    model = GAN(conf,dataset)
    model.run()
