# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/

import numpy as np
import tensorflow as tf

import time, random, threading
import game

from keras.models import *
from keras.layers import *
from keras import backend as K
import sys

#-- constants

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 1
THREAD_DELAY = 0

GAMMA = 0.99

N_STEP_RETURN = 16
GAMMA_N = GAMMA ** N_STEP_RETURN

MIN_BATCH = 64
LEARNING_RATE = 1e-3

LOSS_V = 1            # v loss coefficient
LOSS_ENTROPY = .01     # entropy coefficient

loss_history = []
reward_history = []
frames = 0

CHECKPOINT_DIR = "checkpoints"
#---------
class Brain:
    train_queue = [ [], [], [], [], [] ]    # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, deg, load_checkpoint = True):
        global frames
        frames = 0
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.deg = deg
        self.edge = sum(deg)
        self.num_input = self.edge*3;

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.saver = tf.train.Saver()
        if load_checkpoint:
            checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("checkpoint loaded:", checkpoint.model_checkpoint_path)
                tokens = checkpoint.model_checkpoint_path.split("-")
                # set global step
                frames = int(tokens[1])
                print(">>> global step set: ", frames)
        else:
            print("Could not find old checkpoint")

        self.default_graph.finalize()    # avoid modifications

    def _build_model(self):

        l_input = Input( batch_shape=(None, self.num_input) )
        l_dense1 = Dense(512, activation='relu')(l_input)


        out = []

        #actions
        for e in self.deg:
            out.append(Dense(e, activation='softmax')(l_dense1))
        
        #value
        out.append(Dense(1, activation='linear')(l_dense1))


        model = Model(inputs=[l_input], outputs=out)
        model._make_predict_function()    # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.num_input))
        a_t = tf.placeholder(tf.float32, shape=(None, self.edge))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
        
        pv = model(s_t)
        v = pv.pop()

        p = tf.concat(pv, axis = 1)
        sa = tf.split(a_t,self.deg,1)


        pa = []
        for i in range(len(self.deg)):
            pa.append(tf.reduce_sum(sa[i]*pv[i],axis=1, keep_dims = True))

        pa = tf.concat(pa,axis = 1)

        log_prob = tf.reduce_sum(tf.log(pa + 1e-10) , axis=1, keep_dims = True)
        
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)                                    # maximize policy
        loss_value  = LOSS_V * tf.square(advantage)                                                # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)    # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        gvs = optimizer.compute_gradients(loss_total)
        capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        minimize = optimizer.apply_gradients(capped_gvs)

        #minimize = optimizer.minimize(loss_total)
        return s_t, a_t, r_t, minimize, [loss_total, tf.reduce_mean(r_t), tf.reduce_mean(loss_policy),tf.reduce_mean(loss_value),tf.reduce_mean(entropy),tf.reduce_mean(advantage)]

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)    # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:    # more thread could have passed without lock
                return                                     # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)

        r = r + GAMMA_N * v * s_mask    # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize, loss = self.graph
        _, loss = self.session.run([minimize,loss], feed_dict={s_t: s, a_t: a, r_t: r})

        print("LOSS = ", loss)

        loss_history.append(loss[0])
        reward_history.append(loss[1])
        #for layer in brain.model.layers:
        #    print(layer.get_weights())

    def train_push(self, s, a, r, s_):
        #print("REWARD ",r)
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:    
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            pv = self.model.predict(s)
            v = pv.pop()
            return pv, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            pv = self.model.predict(s)
            pv.pop()
            return pv

    def predict_v(self, s):
        with self.default_graph.as_default():
            pv = self.model.predict(s)        
            return pv[-1]

#---------
printp = False

class Agent:
    def __init__(self,deg):
        self.deg = deg
        self.edge = sum(deg)

        self.memory = []    # used for n_step return
        self.R = 0.

    def act(self, s):
        global frames; frames = frames + 1

        s = np.array([s])
        p = brain.predict_p(s)

        if printp:
            print(p)

        # a = np.argmax(p)
        a = []

        for i in range(len(self.deg)):
            if self.deg[i] == 0:
                a.append(0)
            else:
                a.append(np.random.choice(self.deg[i], p=p[i][0]))

        return a
    
    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _  = memory[0]
            _, _, _, s_ = memory[n-1]

            return s, a, self.R, s_

        a_cats = np.zeros(self.edge)    # turn action into one-hot representation
        
        offset = 0
        for i in range(len(self.deg)):
            a_cats[offset + a[i]] = 1 
            offset += self.deg[i]

        self.memory.append( (s, a_cats, r, s_) )

        #self.R = ( self.R  + r * GAMMA_N) / GAMMA

        #real reward

        self.R = 0

        for i in reversed(range(len(self.memory))):
            self.R = self.R * GAMMA + self.memory[i][2]

        #print("SELF R", self.R)
        #print("REAL R", tmpR)

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = ( self.R - self.memory[0][2] ) / GAMMA
                self.memory.pop(0)        

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = (self.R - self.memory[0][2])
            self.memory.pop(0)    
    
    # possible edge case - if an episode ends in <N steps, the computation is incorrect
        
#---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False):
        threading.Thread.__init__(self)

        self.render = render
        self.env = game.Graph().read("graph.txt",render)
        self.agent = Agent(self.env.deg)

    def reset(self):
        print("RESET!!")
        self.env = game.Graph().read("graph.txt",self.render)
        return self.env.get_state()

    def runEpisode(self):
        s = self.reset()

        R = 0
        Rt = 0

        while True:         
            time.sleep(THREAD_DELAY) # yield 

            #if self.render: self.env.render()

            a = self.agent.act(s)
            r = self.env.next(a)
            s_ = self.env.get_state()

            done = self.env.game_over()

            if done: # terminal state
                s_ = None
                print("GAME OVER")

            self.agent.train(s, a, r, s_)

            s = s_
            R += r
            Rt += r

            if done or self.stop_signal:
                break
            
            if frames % 100 == 0:
                print("Step ", frames)
                print("Reward = ", Rt)
                #reward_history.append(Rt)
                Rt = 0


        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

#---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True

#-- main
env_test = Environment(render=True)

NONE_STATE = np.zeros(sum(env_test.env.deg)*3)

brain = Brain(env_test.env.deg)    # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

print("Animation? ", envs[0].env.display)

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()
# write wall time

brain.saver.save(brain.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = frames)
print("Training finished")

for layer in brain.model.layers:
    print(layer.get_weights())

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()

plt.plot(reward_history)
plt.ylabel('Reward')
plt.xlabel('Batch')
plt.show()

printp = True
env_test.run()

