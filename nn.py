import pandas as pd
import numpy as np
train_mean=0
train_std=0



class nn:
    #nn for classifier works for only one layer
    def __init__(self, no_of_inputs, no_of_outputs, HUs,no_of_layers=1, activation="Sigmoid", dropout=0, weights_seed=0):
        self.no_of_inputs=no_of_inputs
        self.no_of_layers=no_of_layers
        self.no_of_outputs=no_of_outputs
        self.HUs=HUs
        self.act=activation
        self.dropout=dropout
        self.weights_seed=weights_seed
        self.initialize_weights()
    
    def initialize_weights(self):
        np.random.seed(self.weights_seed)
        self.weights=[]
        weights0=0.01 * np.random.randn(self.no_of_inputs+1,self.HUs)
        weights1=0.01 * np.random.randn(self.HUs+1,self.no_of_outputs)
        self.weights.append(weights0);
        self.weights.append(weights1);
    
    def get_weights(self):
        return self.weights
    
    def sigmoid_forward(self,z):
        return 1/(1+np.exp(-z))
        
    def sigmoid_backward(self,z):
        return z*(1-z)
    
    def feed_forward(self, input_arr):
#         print(input_arr.shape)
        self.inps = np.ones((input_arr.shape[0],self.no_of_inputs+1))
        self.inps[:,1:self.no_of_inputs+1]=input_arr
        self.layers=[]
        self.layers.append(self.inps)
        layer0nodes=np.matmul(self.inps,self.weights[0])
        layer0nodes=self.sigmoid_forward(layer0nodes)
        layer0nodes_bias=np.ones((layer0nodes.shape[0],layer0nodes.shape[1]+1))
        layer0nodes_bias[:,1:layer0nodes.shape[1]+1]=layer0nodes
        #output layer nodes
        self.layers.append(layer0nodes_bias)
        layer1nodes=np.matmul(layer0nodes_bias,self.weights[1])
        layer1nodes=self.sigmoid_forward(layer1nodes)
        self.layers.append(layer1nodes)
#         print(len(self.layers))
        return self.layers[-1]
        
    def get_layers(self):
        return self.layers
    
    def compute_err(self,y_hat,y):
        #cross entropy, expects one hot encoded y label
        loss=0
#         print(y_hat)
#         print(y)
        #ylog(y_hat)+(1-y)log(1-y_hat)
        loss_matrix=np.multiply(y,np.log(y_hat))+np.multiply(1-y,np.log(1-y_hat))
        loss=-1*np.sum(loss_matrix)*(1/y.shape[0])
        return loss
    
    def compute_gradients(self,y_hat,y):
        #dE/d(sigma)
        self.gradients=[]
        dE_dsigmaL_init=(-1)*(np.multiply(y,1/y_hat)+np.multiply(1-y,1/(1-y_hat)))
        dE_dsigmaL=np.multiply(dE_dsigmaL_init,y)-np.multiply(dE_dsigmaL_init,1-y)
        
        dE_dsigmaL_dsumL=np.multiply(dE_dsigmaL,self.sigmoid_backward(self.layers[-1]))
        #average
        dE_dsigmaL_dsumL_dw=np.matmul(self.layers[-2].T,dE_dsigmaL_dsumL)*(1/self.layers[-2].shape[0])
        self.gradients.insert(0,dE_dsigmaL_dsumL_dw)
        #update weight with this gradient
        #weights without bias
        weights_wo_bias=self.weights[1][1:,:]
        dE_dsigmaL_dsumL_dsigmal=np.matmul(dE_dsigmaL_dsumL,weights_wo_bias.T)
        layer_wo_bias=self.layers[-2][:,1:]
        dE_dsigmal_dsuml=np.multiply(dE_dsigmaL_dsumL_dsigmal,self.sigmoid_backward(layer_wo_bias))
        dE_dsigmal_dsuml_dw=np.matmul(self.layers[0].T,dE_dsigmal_dsuml)*(1/self.layers[0].shape[0])
        self.gradients.insert(0,dE_dsigmal_dsuml_dw)
        return self.gradients
    
    def update_weights(self, gradients,learning_rate=0.001):
        self.lr=learning_rate
        for i in range(0,len(self.weights)):
            self.weights[i]=self.weights[i]-gradients[i]*learning_rate
    
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
    print(df_norm.head(5))
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

def onehotencoding(y):
    y=y.astype(int)
#     print(y.shape)
    max_y=max(y)
#     max_y=3
    encode_mat=np.zeros((y.shape[0],max_y))
    rows=np.arange(y.shape[0])
    #array([0, 1, 2, 3, 4])
    encode_mat[rows,y-1]=1
    return encode_mat

def train(input_x,y_encod,no_of_epochs=2500,batchSize=100):
    print(input_x.shape)
    print(y_encod.shape)
    no_of_batches=int(input_x.shape[0]/batchSize)
    for i in range(0,no_of_epochs):
        loss=0
        for batch_no in range(0,no_of_batches):
            input_x_train=input_x[batchSize*(batch_no):batchSize*(batch_no+1),:]
            y_encod_train=y_encod[batchSize*(batch_no):batchSize*(batch_no+1),:]
            y_pred=network1.feed_forward(input_x_train)

            loss=loss+network1.compute_err(y_pred,y_encod_train)
        #         print(y_pred)
            grads=network1.compute_gradients(y_pred,y_encod_train)
        #     if(i%1000==0):
        #         print(grads)
        #         print(network1.get_weights())
            network1.update_weights(grads,learning_rate= 0.001)
        if i%100==0:
            loss=loss*1/no_of_batches
            print(loss)

def trainAccuracy(input_x,y_encod):
    y_pred=network1.feed_forward(input_x)
    y_pred_encod=(y_pred == y_pred.max(axis=1)[:,None]).astype(int)
    print(y_pred_encod.shape)
#     print(np.argmax(y_pred_encod,axis=1).shape)
    correct_pred_encod=np.multiply(y_pred_encod,y_encod)
    accuracy=np.sum(correct_pred_encod)/input_x.shape[0]
    print(accuracy)

def testOutput(input_x_test):
    y_pred=network1.feed_forward(input_x_test)
    y_pred_arr=np.argmax(y_pred,axis=1)
    y_pred_arr=y_pred_arr+1
    file=open("submission.csv","w")
    file.write("Id,predicted_class\n")
    print(len(y_pred_arr))
    for i in range(0,len(y_pred_arr)):
        file.write(str(i+1)+","+str(y_pred_arr[i])+"\n")
    file.close()


network1=nn(no_of_inputs=24,no_of_outputs=3,HUs=100)
def main():
    df = pd.read_csv('./data/train.csv')
    train_data=preprocessData(df)
    input_data=train_data.values
    input_x=input_data[:,:-1]
    y_label=input_data[:,-1]
    y_encod=onehotencoding(y_label)
    #training
    train(input_x,y_encod,no_of_epochs=10000)
    trainAccuracy(input_x,y_encod)

    #testing
    df_test = pd.read_csv('./data/test.csv')
    test_data=preprocessData_test(df_test)
    test_x=test_data.values
    testOutput(test_x)




if __name__ == '__main__':
    main()