import tensorflow as tf
import numpy as np
from mnist import MNIST

""" NOTE
Initialize with ,initializer=tf.random_normal_initializer(stddev=1./tf.sqrt(self.conf.dis_hid_num/2.)) can make the convergence faster.
Learning rate shoule be small: 0.00x to avoid NAN problem
"""


dataset = MNIST("./data/MNIST")

graph = tf.Graph()
with graph.as_default():
    z_dimension =  784
    gen_hid_num = 100
    dis_hid_num = 100
    BATCH_SIZE  = 100
    K = 1
    
    x = tf.placeholder(tf.float32,[None,dataset.dimension])
    z = tf.placeholder(tf.float32,[None,z_dimension])

    # graph of Generative part
    with tf.variable_scope("gen") as scope:
        W_gen_enc  = tf.get_variable("W_gen_enc",[z_dimension,gen_hid_num],dtype=tf.float32)
        b_gen_enc  = tf.get_variable("b_gen_enc",[1,gen_hid_num],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        W_gen_dec  = tf.get_variable("W_gen_dec",[gen_hid_num,dataset.dimension],dtype=tf.float32)
        b_gen_dec  = tf.get_variable("b_gen_dec",[1,dataset.dimension],dtype=tf.float32,initializer=tf.constant_initializer(0.0))


    # graph of Discriminative part
    with tf.variable_scope("dis") as scope:                
        W_dis  = tf.get_variable("W_dis",[dataset.dimension,dis_hid_num],dtype=tf.float32)
        b_dis  = tf.get_variable("b_dis",[1,dis_hid_num],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        W_smx  = tf.get_variable("W_smx",[dis_hid_num,1],dtype=tf.float32)
        b_smx  = tf.get_variable("b_smx",[1,1],dtype=tf.float32)

def real():
    return x

def generator():  
    with graph.as_default():
        hg = tf.nn.relu(tf.matmul(z,W_gen_enc) + b_gen_enc)
        f = tf.nn.sigmoid(tf.matmul(hg,W_gen_dec) + b_gen_dec,name="f")# fake samples
        return f

def discriminator(inp):
    with graph.as_default():
        hd   = tf.nn.relu(tf.matmul(inp,W_dis)+b_dis)
        out  = tf.sigmoid(tf.matmul(hd,W_smx)+b_smx)
        return out

#p_fake_is_real = discriminator(input="real")
#p_real_is_real = discriminator(input="fake")

def maximise(p):
    with graph.as_default():
        m = tf.reduce_mean(tf.log(p))
        return m

def minimise(p):
    with graph.as_default():
        m = tf.reduce_mean(tf.log(1-p))
        return m

# cost function of D
#discriminative_objective = maximise(p_real_is_real) + minimise(p_fake_is_real)#-tf.reduce_mean(tf.log(out_x)) - tf.reduce_mean(tf.log(1-out_f))
# cost function of G
#generative_objective = maximise(p_fake_is_real)#-tf.reduce_mean(tf.log(out_f))


def train(gen_cost,dis_cost,learning_rate=0.0001):
    with graph.as_default():
        # construct model
        #gen_cost,dis_cost,f = self.build_model()
        # training
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gen_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"gen")
        #print(gen_vars)
        f = tf.get_default_graph().get_tensor_by_name("f:0")
        gen_op = optimizer.minimize(-gen_cost,var_list = gen_vars)
        
        dis_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"dis")
        dis_op = optimizer.minimize(-dis_cost,var_list = dis_vars)
        
        init  = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)
            
        iter_num = 0
        total_dis_err = total_gen_err = 0
        while True:
            dis_errs = 0
            for k in range(K):
                # sample from data
                x_ = dataset.samples(batch_size=BATCH_SIZE)
                z_ = zsample(BATCH_SIZE,z_dimension)

                # Update discriminator
                _,dis_err = session.run([dis_op,dis_cost],{x:x_,z:z_})
                dis_errs+= dis_err
            total_dis_err += dis_errs/K
            # Update generator
            z_ = zsample(BATCH_SIZE,z_dimension)
            _,gen_err = session.run([gen_op,gen_cost],{z:z_})
            total_gen_err += gen_err
                
                
            iter_num +=1
            if iter_num%100==0:
                print(iter_num,total_dis_err,total_gen_err)
                total_dis_err = total_gen_err = 0
                if iter_num%500==0:
                    images = session.run(f,{z:zsample(25,z_dimension)})
                    dataset.show_samples(images)

def zsample(m, n):
    #return np.random.uniform(size=[m, n])
    #return np.random.uniform(-1.,1.,size=[m, n])
    return np.random.normal(scale=1,size=[m,n])



if __name__=="__main__":
    fake_images = generator()
    real_images = real()
    p_fake_is_real = discriminator(fake_images)
    p_real_is_real = discriminator(real_images)
    discriminator_objective = maximise(p_real_is_real) + minimise(p_fake_is_real)
    generator_objective = maximise(p_fake_is_real)
    train(generator_objective,discriminator_objective)