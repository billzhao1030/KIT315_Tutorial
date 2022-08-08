import tensorflow as tf
import numpy as np
"""
GAN - Ian Goodfellow et. al. 2014
Implemented dy Son .N Tran & Son Vu
"""
""" NOTE
Initialize with ,initializer=tf.random_normal_initializer(stddev=1./tf.sqrt(self.conf.dis_hid_num/2.)) can make the convergence faster.
Learning rate shoule be small: 0.00x to avoid NAN problem

"""
class GAN(object):
    def __init__(self,conf,dataset):
        self.conf = conf
        self.dataset = dataset

    def build_model(self):
        # Create places holder
        self.x = tf.placeholder(tf.float32,[None,self.dataset.dimension])
        self.z = tf.placeholder(tf.float32,[None,self.conf.z_dimension])
        # graph of Generative part
        with tf.variable_scope("gen") as scope:
            W_gen_enc  = tf.get_variable("W_gen_enc",[self.conf.z_dimension,self.conf.gen_hid_num],dtype=tf.float32,initializer=tf.random_normal_initializer(stddev=1./tf.sqrt(self.conf.gen_hid_num/2.)))
            b_gen_enc  = tf.get_variable("b_gen_enc",[1,self.conf.gen_hid_num],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            W_gen_dec  = tf.get_variable("W_gen_dec",[self.conf.gen_hid_num,self.dataset.dimension],dtype=tf.float32)
            b_gen_dec  = tf.get_variable("b_gen_dec",[1,self.dataset.dimension],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        
        hg = tf.nn.relu(tf.matmul(self.z,W_gen_enc) + b_gen_enc)
        f = tf.nn.sigmoid(tf.matmul(hg,W_gen_dec) + b_gen_dec)# fake samples
        # graph of Discriminative part
        with tf.variable_scope("dis") as scope:                
            W_dis  = tf.get_variable("W_dis",[self.dataset.dimension,self.conf.dis_hid_num],dtype=tf.float32,initializer=tf.random_normal_initializer(stddev=1./tf.sqrt(self.conf.dis_hid_num/2.)))
            b_dis  = tf.get_variable("b_dis",[1,self.conf.dis_hid_num],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            W_smx  = tf.get_variable("W_smx",[self.conf.dis_hid_num,1],dtype=tf.float32)
            b_smx  = tf.get_variable("b_smx",[1,1],dtype=tf.float32)
            
        hd_x   = tf.nn.relu(tf.matmul(self.x,W_dis)+b_dis)
        hd_f   = tf.nn.relu(tf.matmul(f,W_dis)+b_dis)
        out_x  = tf.sigmoid(tf.matmul(hd_x,W_smx)+b_smx)
        out_f  = tf.sigmoid(tf.matmul(hd_f,W_smx)+b_smx)
        # cost function of D
        dis_cost = -tf.reduce_mean(tf.log(out_x)) - tf.reduce_mean(tf.log(1-out_f))
        # cost function of G
        gen_cost = -tf.reduce_mean(tf.log(out_f))
        # return the generative cost, discriminative cost
        # and generator of the model
        return gen_cost,dis_cost,f
    def run(self):
        with tf.Graph().as_default():
            # construct model
            gen_cost,dis_cost,f = self.build_model()
            # training
            optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.lr)
            gen_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"gen")
            gen_op = optimizer.minimize(gen_cost,var_list = gen_vars)
            
            dis_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"dis")
            dis_op = optimizer.minimize(dis_cost,var_list = dis_vars)

            init  = tf.global_variables_initializer()
            session = tf.Session()
            session.run(init)
            
            iter_num = 0
            total_dis_err = total_gen_err = 0
            while True:
                dis_errs = 0
                for k in range(self.conf.k):
                    # sample from data
                    x = self.dataset.samples(batch_size=self.conf.batch_size)
                    z = zsample(self.conf.batch_size,self.conf.z_dimension)

                    # Update discriminator
                    _,dis_err = session.run([dis_op,dis_cost],{self.x:x,self.z:z})
                    dis_errs+= dis_err
                total_dis_err += dis_errs/self.conf.k
                # Update generator
                z = zsample(self.conf.batch_size,self.conf.z_dimension)
                _,gen_err = session.run([gen_op,gen_cost],{self.z:z})
                total_gen_err += gen_err
                
                
                iter_num +=1
                if iter_num%100==0:
                    print(total_dis_err,total_gen_err)
                    total_dis_err = total_gen_err = 0
                    print("plotting ...")
                    images = session.run(f,{self.z:zsample(25,self.conf.z_dimension)})
                    self.dataset.show_samples(images)

            
            

def zsample(m, n):
    #return np.random.uniform(size=[m, n])
    #return np.random.uniform(-1.,1.,size=[m, n])
    return np.random.normal(scale=1,size=[m,n])
