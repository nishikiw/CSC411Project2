from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import scipy.stats

ws = []

def main():
    theta = array([-3, 1.5])
    N = 50          # Training set size - number of outliers
    N_test = 30     # Test set size
    outliers = 5    # Outliers in training set
    sigma = 100
    gen_lin_data_1d(theta, N, outliers, N_test, sigma)


def f_project1(x, y, theta):
    """Cost function for part 6."""
    
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (dot(theta.T,x) - y) ** 2)


def df_project1(x, y, theta):
    """Derivative of the cost function."""
    
    x = vstack( (ones((1, x.shape[1])), x))
    return 2 * dot(x, (dot(theta.T, x) - y).T)
  

def f_project2(x, y, w):
    x = vstack( (ones((1, x.shape[1])), x)) # add the "b"
    p = lin_combin(w, x)
    return -sum(y*log(p)) 
    
    
def df_project2(x, y, w):
    x = vstack( (ones((1, x.shape[1])), x)) # add the "b"
    p = lin_combin(w, x)
    return dot(x, (p - y).T)


def lin_combin(w, x):
    o = dot(w.T, x)
    return softmax(o)
    
    
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def grad_descent(f, df, x, y, init_t, alpha, max_iter):
    """
    Gradient descent function borrowed from
    http://www.cs.toronto.edu/~guerzhoy/411/lec/W02/python/linreg.html
    """
    
    global ws
    ws = []
    
    EPS = 1e-8   #EPS = 10**(-8)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        # if iter % 500 == 0:
        #     print "Iter", iter
        #     print "f(x) = %.2f" % (f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        if iter % 1000 == 0:
            ws.append(t.copy())
        iter += 1
    return t


def plot_line(theta, x_min, x_max, color, label):
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((    ones_like(x_grid_raw),
                         x_grid_raw,
                    ))
    y_grid = dot(theta, x_grid)
    plot(x_grid[1,:], y_grid, color, label=label)


def test_performance(x, theta, y):
    correct = 0
    x = vstack( (ones((1, x.shape[1])), x))
    result = dot(theta.T, x)
    for i in range(0, y.shape[1]):
        if y[argmax(result[:, i]), i] == 1:
            correct += 1
    return correct * 100 / y.shape[1]
    

def gen_lin_data_1d(theta, N, outliers, N_test, sigma):

    # Generate data set.
    # General training data
    random.seed(0)
    x1_general = 100*(random.random((N))-.5)
    random.seed(1)
    x2_general = 100*(random.random((N))-.5)
    
    # Outliers
    random.seed(2)
    x1_outlier = 100+100*random.random((outliers))
    random.seed(3)
    x2_outlier = 100+100*random.random((outliers))
    
    x1 = concatenate((x1_general, x1_outlier))
    x2 = concatenate((x2_general, x2_outlier))
    
    # Set x
    x = vstack((x1, x2))
    
    # Set y
    y = zeros((2, len(x1)))
    colors = array([])
    color1 = array([1., 0., 0., 1])
    color2 = array([0., 1., 0., 1])
    
    # Get y: if x2 is above x-axis, y is class 1. Else, y is class 2.
    for i in range(0, len(x1)):
        if x2[i] > 0:
            y[0, i] = 1
            y[1, i] = 0
            if len(colors) == 0:
                colors = color1
            else:
                colors = vstack((colors, color1))
        else:
            y[0, i] = 0
            y[1, i] = 1
            if len(colors) == 0:
                colors = color2
            else:
                colors = vstack((colors, color2))        
    
    figure(1)
    # Plot actual data
    scatter(x1, x2, s=1, color=colors, label="training set")
    
    # Plot the boundary of two classes.
    axhline(y=0, color='k', label="boundary")
    
    legend(loc = 'upper center')
    xlim([-200, 200])
    ylim([-200, 200])
    savefig("part5_training_data.png")
    
    random.seed(5)
    init_t = random.normal(0.0, 1.0, (x.shape[0]+1, y.shape[0]))/math.sqrt((x.shape[0]+1) * y.shape[0])
    max_iter = 30000
    alpha1 = 1e-6
    
    # Using project 1's model (linear regression) to test performance on dataset.
    theta_p1 = grad_descent(f_project1, df_project1, x, y, init_t, alpha1, max_iter)
    
    project1_performance = []
    # Plot project 1 performance
    for i in range (0, len(ws)):
        acc = test_performance(x, ws[i], y)
        project1_performance.append(acc)
    
    # Using project 2's model (logistic regression) to test performance on dataset.
    alpha2 = 1e-3
    theta_p2 = grad_descent(f_project2, df_project2, x, y, init_t, alpha2, max_iter)
    
    project2_performance = []
    # Plot project 2 performance
    for i in range (0, len(ws)):
        acc = test_performance(x, ws[i], y)
        project2_performance.append(acc)
    
    x_axis = []
    for i in range(0, 30001, 1000):
        x_axis.append(i)
        
    figure(2)
    ylim(0,110)
    plot(x_axis, project1_performance, label="linear regression")
    plot(x_axis, project2_performance, label="logistic regression")
    legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    xlabel('Iteration')
    ylabel('Correctness(%)')
    savefig('part5_learning_curve.png')
    
    
    # Test performance
    random.seed(6)
    x1_test = 400*(random.random((N_test))-.5)
    random.seed(7)
    x2_test = 400*(random.random((N_test))-.5)
    
    # Set x
    x_test = vstack((x1_test, x2_test))
    
    # Set y
    colors_test = array([])
    y_test = zeros((2, len(x1_test)))
    for i in range(0, len(x1_test)):
        if x2_test[i] > 0:
            y_test[0, i] = 1
            y_test[1, i] = 0
            if len(colors_test) == 0:
                colors_test = color1
            else:
                colors_test = vstack((colors_test, color1))
        else:
            y_test[0, i] = 0
            y_test[1, i] = 1
            if len(colors_test) == 0:
                colors_test = color2
            else:
                colors_test = vstack((colors_test, color2))
            
    performance_p1 = test_performance(x_test, theta_p1, y_test)
    print("Project 1 performance is "+str(performance_p1)+"%")
    performance_p2 = test_performance(x_test, theta_p2, y_test)
    print("Project 2 performance is "+str(performance_p2)+"%")
    
    figure(3)
    # Plot test data
    scatter(x1_test, x2_test, s=1, color=colors_test, label="test set")
    
    # Plot the boundary of two classes.
    axhline(y=0, color='k', label="boundary")
    
    legend(loc = 'upper center')
    xlim([-200, 200])
    ylim([-200, 200])
    savefig("part5_test_data.png")
    

if __name__ == '__main__':
    main()