import pandas as pd
import numpy as np
train_mean=0
train_std=0



import numpy as np
class nn:
    #nn for classifier works for only one layer
    def __init__(self, no_of_inputs, no_of_outputs, HUs,no_of_hidden_layers=1, activation=["sigmoid","sigmoid"],regu=0, dropout=0, weights_seed=0):
        
        
        
        self.no_of_inputs=no_of_inputs
        self.no_of_hidden_layers=no_of_hidden_layers
        self.no_of_outputs=no_of_outputs
        self.HUs=HUs #hidden units array
        self.act=activation
        self.dropout=dropout
        self.weights_seed=weights_seed
        self.regu=regu
        #sanity checks
        if(len(self.HUs) != self.no_of_hidden_layers):
            print("error mismatch hidden units and layers")
        if((self.no_of_hidden_layers+1)!=len(activation)):
            print("error mismatch activations and layers")
        self.initialize_weights()
    
    def initialize_weights(self):
        
        np.random.seed(self.weights_seed)
        self.weights=[]
        weights0=0.01 * np.random.randn(self.no_of_inputs+1,self.HUs[0])
        self.weights.append(weights0);
        # to add multiple layers
        for i in range(1,self.no_of_hidden_layers):
            weightsh=0.01 * np.random.randn(self.HUs[i-1]+1,self.HUs[i])
            self.weights.append(np.copy(weightsh));
              
            
        weightsl=0.01 * np.random.randn(self.HUs[-1]+1,self.no_of_outputs)
        self.weights.append(weightsl);
    
    def get_weights(self):
        return self.weights
    
    def sigmoid_forward(self,z):
#         print("sig forward")
        return 1/(1+np.exp(-z))
        
    def sigmoid_backward(self,z):
#         print("sig bckward")
        return z*(1-z)
    
    def tanh_forward(self,z):
#         print("tanh forwad")
        return np.tanh(z) 
    
    def tanh_backward(self,z):
#         print("tanh backward")
        return 1-z**2
    
    #not working relu
    def relu_forward(self,z):
#         print("relu forwad")
        return z * (z > 0) 
    
    def relu_backward(self,z):
#         print("relu backward")
        
        return 1 * (z > 0)
    
    def softplus_forward(self,z):
        return np.log(1+np.exp(z)) 
    
    def softplus_backward(self,z): 
        return 1-1/np.exp(z)
    
    def activation_forward(self,z,act_index):
        if(self.act[act_index]=="sigmoid"):
            return self.sigmoid_forward(z)
        elif (self.act[act_index]=="softplus"):
            return self.softplus_forward(z)
        elif (self.act[act_index]=="tanh"):
            return self.tanh_forward(z)
        elif (self.act[act_index]=="relu"):
            return self.relu_forward(z)
        else:
            print("error invalid activation")
    
    def activation_backward(self,z,act_index):
        if(self.act[act_index]=="sigmoid"):
            return self.sigmoid_backward(z)
        elif (self.act[act_index]=="softplus"):
            return self.softplus_backward(z)
        elif (self.act[act_index]=="tanh"):
            return self.tanh_backward(z)
        elif (self.act[act_index]=="relu"):
            return self.relu_backward(z)
        else:
            print("error invalid activation")
    
    def feed_forward(self, input_arr):
#         print(input_arr.shape)
        self.inps = np.ones((input_arr.shape[0],self.no_of_inputs+1))
        self.inps[:,1:self.no_of_inputs+1]=input_arr
        self.layers=[]
        self.layers.append(self.inps)
        layer0nodes=np.matmul(self.inps,self.weights[0])
        layer0nodes=self.activation_forward(layer0nodes,0)
        layer0nodes_bias=np.ones((layer0nodes.shape[0],layer0nodes.shape[1]+1))
        layer0nodes_bias[:,1:layer0nodes.shape[1]+1]=layer0nodes
        #for all the internal layer nodes
        self.layers.append(layer0nodes_bias)
        for i in range(1,self.no_of_hidden_layers):
            layerinodes=np.matmul(self.layers[-1],self.weights[i])
            layerinodes=self.activation_forward(layerinodes,i)
            layerinodes_bias=np.ones((layerinodes.shape[0],layerinodes.shape[1]+1))
            layerinodes_bias[:,1:layerinodes.shape[1]+1]=layerinodes
            self.layers.append(np.copy(layerinodes_bias))
            
        #for the last layer
        layer1nodes=np.matmul(self.layers[-1],self.weights[-1])
        layer1nodes=self.activation_forward(layer1nodes,-1)
        self.layers.append(layer1nodes)
#         print(len(self.layers))
        return self.layers[-1]
        
    def get_layers(self):
        return self.layers
    
    #can use with sigmoid
    def compute_cross_entropy_err(self,y_hat,y):
        #cross entropy, expects one hot encoded y label
        loss=0
        #ylog(y_hat)+(1-y)log(1-y_hat)
        loss_matrix=np.multiply(y,np.log(y_hat))+np.multiply(1-y,np.log(1-y_hat))
        loss_matrix=np.nan_to_num(loss_matrix)
        #computing sum of squares of weights
        sum_square=0
        for i in range(0,len(self.weights)):
            sum_square=sum_square+np.sum((np.array(self.weights[i]))**2)
        
        loss=-1*np.sum(loss_matrix)*(1/y.shape[0])+self.regu*(1/y.shape[0])*sum_square
        return loss
    
    def compute_gradients(self,y_hat,y):
        #dE/d(sigma)
        self.gradients=[]
        dE_dsigmaL=(-1)*(np.multiply(y,1/y_hat)-np.multiply(1-y,1/(1-y_hat)))
#         dE_dsigmaL=np.multiply(dE_dsigmaL_init,y)-np.multiply(dE_dsigmaL_init,1-y)
        dE_dsigmaL=np.nan_to_num(dE_dsigmaL)
        dE_dsigmaL_dsumL=np.multiply(dE_dsigmaL,self.activation_backward(self.layers[-1],-1))
        dE_dsigmaL_dsumL=np.nan_to_num(dE_dsigmaL_dsumL)
        #average
        #gradient for the last layer
        #update weight with this gradient
        dE_dsigmaL_dsumL_dw=np.matmul(self.layers[-2].T,dE_dsigmaL_dsumL)*(1/self.layers[-2].shape[0])
        self.gradients.insert(0,dE_dsigmaL_dsumL_dw)
        
        for i in range(self.no_of_hidden_layers,0,-1):
            #weights without bias
            weights_wo_bias=self.weights[i][1:,:]
            dE_dsigmaL_dsumL_dsigmal=np.matmul(dE_dsigmaL_dsumL,weights_wo_bias.T)
            layer_wo_bias=self.layers[i][:,1:]
            dE_dsigmaL_dsumL=np.multiply(dE_dsigmaL_dsumL_dsigmal,self.activation_backward(layer_wo_bias,i-1))
            dE_dsigmaL_dsumL=np.nan_to_num(dE_dsigmaL_dsumL)
            dE_dsigmal_dsuml_dw=np.matmul(self.layers[i-1].T,dE_dsigmaL_dsumL)*(1/self.layers[i-1].shape[0])
            self.gradients.insert(0,np.copy(dE_dsigmal_dsuml_dw))
        
        return self.gradients
    
    def update_weights(self, gradients,learning_rate=0.001):
        self.lr=learning_rate
        for i in range(0,len(self.weights)):
            self.weights[i]=np.copy(self.weights[i])-gradients[i]*learning_rate-self.regu*learning_rate*np.copy(self.weights[i])
    
    #dont use predict for now
    def predict(self,input_arr):
        inps = np.ones((input_arr.shape[0],self.no_of_inputs+1))
        inps[:,1:self.no_of_inputs+1]=input_arr
        self.layers=[]
#         self.layers.append(self.inps)
        layer0nodes=np.matmul(self.inps,self.weights[0])
        layer0nodes=self.sigmoid_forward(layer0nodes)
        layer0nodes_bias=np.ones((layer0nodes.shape[0],layer0nodes.shape[1]+1))
        layer0nodes_bias[:,1:layer0nodes.shape[1]+1]=layer0nodes
        #output layer nodes
#         self.layers.append(layer0nodes_bias)
        layer1nodes=np.matmul(layer0nodes_bias,self.weights[1])
        layer1nodes=self.sigmoid_forward(layer1nodes)
#         self.layers.append(layer1nodes)
#         print(len(self.layers))
        return layer1nodes
        
        
               

def preprocessData(df):
    df['post_day'] = df['post_day'].factorize(sort=True)[0]
    df['basetime_day'] = df['basetime_day'].factorize(sort=True)[0]
#     df=df.head(5)
    # print(df)
    df = df.sample(frac=1,random_state=1).reset_index(drop=True)
    df_norm=df.drop(labels='target', axis=1)
    print(df_norm.mean())
    print(df_norm.std())
    df_norm=(df_norm-df_norm.mean())/df_norm.std()
    global train_mean
    global train_std
    train_mean=df_norm.mean()
    train_std=df_norm.std()
#     df_norm=(df_norm-df_norm.min())/(df_norm.max()-df_norm.min())
    df_norm.fillna(0,inplace=True)
    # print(df_norm.head(5))
    df_norm=pd.concat([df_norm, df['target']], axis=1)
#     df_norm=df_norm[['page_likes','daily_crowd','target']]
    return df_norm

def preprocessData_test(df):
    df['post_day'] = df['post_day'].factorize(sort=True)[0]
    df['basetime_day'] = df['basetime_day'].factorize(sort=True)[0]
#     df=df.head(5)
    df_norm=(df-train_mean)/train_std
    df_norm.fillna(0,inplace=True)
    return df_norm

def onehotencoding(y,max_y):
    y=y.astype(int)
#     print(y.shape)
#     max_y=3
    encode_mat=np.zeros((y.shape[0],max_y))
    rows=np.arange(y.shape[0])
    #array([0, 1, 2, 3, 4])
    encode_mat[rows,y-1]=1
    return encode_mat

def train(network,input_x,y_encod,logfile,lr=0.001,no_of_epochs=2500,batchSize=100,do_validate=0,val_x=0,y_val_encod=0):
    # print(input_x.shape)
    # print(y_encod.shape)
    no_of_batches=int(input_x.shape[0]/batchSize)
    val_loss_arr=np.array([])
    file=open(logfile,"w")
    file.write("Epoch no,Lr,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n")
    np.random.seed(0)
    for i in range(0,no_of_epochs):
        train_loss=0
        val_loss=0
        train_acc=0
        val_acc=0
        for batch_no in range(0,no_of_batches):
            idx = np.random.randint(input_x.shape[0], size=batchSize)
            input_x_train=input_x[idx,:]
            y_encod_train=y_encod[idx,:]
            y_pred=network.feed_forward(input_x_train)

            train_loss=train_loss+network.compute_cross_entropy_err(y_pred,y_encod_train)
        #         print(y_pred)
            grads=network.compute_gradients(y_pred,y_encod_train)
        #     if(i%1000==0):
        #         print(grads)
        #         print(network1.get_weights())
            network.update_weights(grads,learning_rate= lr)
        
        if do_validate==1 and ((i+1)%100==0 or i==0):
            y_pred_val=network.feed_forward(val_x)
            val_loss=network.compute_cross_entropy_err(y_pred_val,y_val_encod)
            print("val loss: "+str(val_loss))
            
            #validation accuracy
            val_acc=trainAccuracy(network,val_x,y_val_encod)
            print("val Accu: "+str(val_acc))

            #lr changing logic
            val_loss_arr=np.append(val_loss_arr,val_loss)
            if(val_loss_arr.shape[0]>=5):
                std_val=np.std(val_loss_arr[-5:])
                if(std_val<0.004):
                    lr=lr/4
                    val_loss_arr=np.array([])
        if ((i+1)%100==0 or i==0):
            train_loss=train_loss*1/no_of_batches
            print("train loss: "+str(train_loss))
            #train accuracy
            train_acc=trainAccuracy(network,input_x,y_encod)
            print("train Accu: "+str(train_acc))

            file.write(str(i+1)+","+str(lr)+","+str(train_loss)+","+str(train_acc)+","+str(val_loss)+","+str(val_acc)+"\n")
        if(lr<0.001):
            break
    file.close()

def trainAccuracy(network,input_x,y_encod):
    y_pred=network.feed_forward(input_x)
    y_pred_encod=(y_pred == y_pred.max(axis=1)[:,None]).astype(int)
    # print(y_pred_encod.shape)
#     print(np.argmax(y_pred_encod,axis=1).shape)
    correct_pred_encod=np.multiply(y_pred_encod,y_encod)
    accuracy=np.sum(correct_pred_encod)/input_x.shape[0]
    return accuracy

def testOutput(network,input_x_test,out_file):
    y_pred=network.feed_forward(input_x_test)
    y_pred_arr=np.argmax(y_pred,axis=1)
    y_pred_arr=y_pred_arr+1
    file=open(out_file,"w")
    file.write("Id,predicted_class\n")
    # print(len(y_pred_arr))
    for i in range(0,len(y_pred_arr)):
        file.write(str(i+1)+","+str(y_pred_arr[i])+"\n")
    file.close()



def main(taskno):
    #training data
    df = pd.read_csv('./../data/train.csv')
    train_data=preprocessData(df)

    #testing data
    df_test = pd.read_csv('./../data/test.csv')
    test_data=preprocessData_test(df_test)

    max_y=3 #max no of labels

    if(taskno==2):
        network1=nn(no_of_inputs=24,no_of_outputs=3,HUs=[100],no_of_hidden_layers=1, activation=["sigmoid","sigmoid"])
        input_data=train_data.values
        input_x=input_data[:,:-1]
        y_label=input_data[:,-1]
        y_encod=onehotencoding(y_label,max_y)
        #training
        train(network1,input_x,y_encod,"log/task2_log.csv",no_of_epochs=5000)
        # trainAccuracy(input_x,y_encod)
        
        #testing
        test_x=test_data.values
        testOutput(network1,test_x,"log/submission_task2.csv")
    
    elif(taskno==3):
        val_percent=0.1
        val_size=int(val_percent*len(train_data.index))
        val_data=train_data.tail(val_size)
        input_data_train=train_data.head(len(train_data.index)-val_size)
        input_data=input_data_train.values
        #train data
        input_x=input_data[:,:-1]
        y_label=input_data[:,-1]
        y_encod=onehotencoding(y_label,max_y)
        
        #val data
        val_input=val_data.values
        val_x=val_input[:,:-1]
        y_val_label=val_input[:,-1]
        y_val_encod=onehotencoding(y_val_label,max_y)


        network1=nn(no_of_inputs=24,no_of_outputs=3,HUs=[100,100],regu=0.01,no_of_hidden_layers=2, activation=["sigmoid","sigmoid","sigmoid"])
        
        train(network1,input_x,y_encod,"log/task3_log_1.csv",lr=0.1,no_of_epochs=6000,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network1,test_x,"log/submission_task3_1.csv")


        network2=nn(no_of_inputs=24,no_of_outputs=3,HUs=[100],regu=0.01,no_of_hidden_layers=1, activation=["sigmoid","sigmoid"])
        
        train(network2,input_x,y_encod,"log/task3_log_2.csv",lr=0.1,no_of_epochs=6000,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network2,test_x,"log/submission_task3_2.csv")


        network3=nn(no_of_inputs=24,no_of_outputs=3,HUs=[48,96,48],regu=0.001,no_of_hidden_layers=3, activation=["sigmoid","sigmoid","sigmoid","sigmoid"])
        
        train(network3,input_x,y_encod,"log/task3_log_3.csv",lr=0.1,no_of_epochs=6000,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network3,test_x,"log/submission_task3_3.csv")



        network4=nn(no_of_inputs=24,no_of_outputs=3,HUs=[96,48,96],regu=0.0001,no_of_hidden_layers=3, activation=["sigmoid","sigmoid","sigmoid","sigmoid"])
        
        train(network4,input_x,y_encod,"log/task3_log_4.csv",lr=0.1,no_of_epochs=6000,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network4,test_x,"log/submission_task3_4.csv")



        network5=nn(no_of_inputs=24,no_of_outputs=3,HUs=[48,96,48],regu=0.0001,no_of_hidden_layers=3, activation=["sigmoid","sigmoid","sigmoid","sigmoid"])
        
        train(network5,input_x,y_encod,"log/task3_log_5.csv",lr=0.1,no_of_epochs=6000,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network5,test_x,"log/submission_task3_5.csv")


    elif(taskno==4):

        val_percent=0.1
        val_size=int(val_percent*len(train_data.index))
        val_data=train_data.tail(val_size)
        input_data_train=train_data.head(len(train_data.index)-val_size)
        input_data=input_data_train.values
        #train data
        input_x=input_data[:,:-1]
        y_label=input_data[:,-1]
        y_encod=onehotencoding(y_label,max_y)
        
        #val data
        val_input=val_data.values
        val_x=val_input[:,:-1]
        y_val_label=val_input[:,-1]
        y_val_encod=onehotencoding(y_val_label,max_y)

        network1=nn(no_of_inputs=24,no_of_outputs=3,HUs=[100],regu=0,no_of_hidden_layers=1, activation=["relu","sigmoid"])
        
        train(network1,input_x,y_encod,"log/task4_log_1.csv",lr=0.01,no_of_epochs=300,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network1,test_x,"log/submission_task4_1.csv")

        network2=nn(no_of_inputs=24,no_of_outputs=3,HUs=[100],regu=0,no_of_hidden_layers=1, activation=["tanh","sigmoid"])
        
        train(network2,input_x,y_encod,"log/task4_log_2.csv",lr=0.01,no_of_epochs=2000,do_validate=1,val_x=val_x,y_val_encod=y_val_encod)

        #testing
        test_x=test_data.values
        testOutput(network2,test_x,"log/submission_task4_2.csv")
        
    elif(taskno==5):
        pass





if __name__ == '__main__':
    main(4)