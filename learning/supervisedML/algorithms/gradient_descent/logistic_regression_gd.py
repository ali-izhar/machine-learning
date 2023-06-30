import numpy as np



def compute_cost(x, y, w, b):
    """
    Computes the squared error cost function for linear regression.
    """
    m = x.shape[0]
    cost = (1/(2*m)) * np.sum((np.dot(x, w) + b - y)**2)
    return cost
    

def compute_gradient(x, y, w, b):
    """
    Computes the gradient of the squared error cost function for linear regression.
    Args:
        x (ndarray (m, )): Data, m examples
        y (ndarray (m, )): Labels or targets
        w, b (scalar): Parameters of the model
    Returns:
        dw, db (scalar): Gradients of the cost function
    """
    m = x.shape[0]
    dw = (1/m) * np.sum((np.dot(x, w) + b - y) * x)
    db = (1/m) * np.sum(np.dot(x, w) + b - y)



    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        prediction = np.dot(X,theta)
        theta = theta -(1/m)*learning_rate*(X.T.dot((prediction - y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = compute_cost(X,y,theta)
    return theta, cost_history, theta_history

# stochastic gradient descent
def stochastic_gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)
            theta = theta -(1/m)*learning_rate*(X_i.T.dot((prediction - y_i)))
            cost += compute_cost(X_i, y_i, theta)
        cost_history[it]  = cost
    return theta, cost_history

# mini-batch gradient descent
def minibatch_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10,batch_size =20):
    m = len(y)
    cost_history = np.zeros(iterations)
    n_batches = int(m/batch_size)
    for it in range(iterations):
        cost =0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0,m,batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            X_i = np.c_[np.ones(len(X_i)),X_i]
            prediction = np.dot(X_i,theta)
            theta = theta -(1/m)*learning_rate*(X_i.T.dot((prediction - y_i)))
            cost += compute_cost(X_i, y_i, theta)
        cost_history[it]  = cost
    return theta, cost_history
